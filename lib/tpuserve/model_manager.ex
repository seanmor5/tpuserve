defmodule TPUServe.ModelManager do
  @moduledoc """
  Manages which models are exposed by the server. The TPUServe
  Model struct just wraps the underlying TPUServeModel NIF resource
  which just wraps TPU Loaded Program Handles and pre-allocated input
  and output buffers (LOL).

  The Model Manager handles requests for access to underlying model
  resources. The model manager needs to ensure that only one process
  can access the underlying resource at once. We achieve this with a
  lock; however, in order to try and squeeze more performance out of
  the server, we introduce an additional state to support autobatching.

  TPUs will always implicitly pad inputs to have batch or feature sizes
  of 128, which means we are really wasting resources when sending requests
  at batch size 1. TPUs are not meant for latency-sensitive applications.
  A sort of way to augment this performance is to introduce a temporary
  waiting state which waits for simulataneous requests to the same model
  resources and sends those requests as 1 to the TPU.
  """

  alias TPUServe.{Driver, Model, ModelConfig}
  require Logger
  use GenServer

  @model_file "model.hlo.txt"
  @config_file "config.json"

  # TODO: Make model thread safe

  def start_link(repo, opts \\ []) do
    GenServer.start_link(__MODULE__, repo, name: __MODULE__)
  end

  def fetch(endpoint) do
    GenServer.call(__MODULE__, {:fetch, endpoint})
  end

  def init(repo) do
    model_paths =
      repo
      |> File.ls!()
      |> Enum.map(&Path.join(repo, &1))
      |> Enum.filter(&File.dir?/1)

    case try_load_models(model_paths) do
      models when models == %{} ->
        Logger.warn("Manager did not successfully load any models")
        {:ok, %{}}

      models ->
        endpoints = Map.keys(models)
        Logger.info("Successfully loaded models for endpoints #{inspect(endpoints)}")
        {:ok, models}
    end
  end

  defp try_load_models(model_paths) do
    # TODO: Error?
    driver = Driver.fetch!()

    model_paths
    |> Enum.map(fn path ->
      with {:ok, {endpoint, config_file, model_file}} <- get_model_files(path),
           {:ok, config} <- File.read(config_file),
           # TODO: Make this {:ok, ...}
           %ModelConfig{} = model_config <- ModelConfig.parse!(config),
           {:ok, %Model{} = model} <- Model.load(driver, model_file, model_config) do
        {:ok, {endpoint, model}}
      else
        # TODO: Match error
        _ ->
          Logger.error("Failed to load model")
          {:error, :bad}
      end
    end)
    |> Enum.filter(fn
      {:ok, _} -> true
      {:error, _} -> false
    end)
    |> Map.new(fn {:ok, {k, v}} -> {k, v} end)
  end

  defp get_model_files(path) do
    endpoint = Path.basename(path)

    # TODO: Add verbose error for both failure paths
    config_file = Path.join(path, @config_file)
    model_file = Path.join(path, @model_file)

    cond do
      not File.exists?(model_file) ->
        {:error, :missing_model_file}

      not File.exists?(config_file) ->
        {:error, :missing_config_file}

      true ->
        {:ok, {endpoint, config_file, model_file}}
    end
  end

  def handle_call({:fetch, model}, _from, state) do
    case state[model] do
      nil ->
        {:reply, {:error, :not_found}, state}

      model_ref ->
        {:reply, {:ok, model_ref}, state}
    end
  end
end
