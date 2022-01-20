defmodule TPUServe.ModelManager do
  @moduledoc """
  Manages which models are exposed by the server.

  Stores TPUServeModel references in a map of
  `%{endpoint => model_ref}`. TPUServeModels are just
  loaded program handles and input/output buffer handles.
  """

  alias TPUServe.Model
  alias TPUServe.ModelConfig
  require Logger
  use GenServer

  @model_file "model.hlo.txt"
  @config_file "config.json"

  # TODO: Make model thread safe

  def init(repo) do
    model_paths =
      repo
      |> File.ls!()
      |> Enum.filter(&File.dir?(Path.join(repo, &1)))

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

  def fetch_model(model) do
    GenServer.call(__MODULE__, {:fetch, model})
  end

  defp try_load_models(model_paths) do
    # TODO: Error?
    driver = TPUServe.Driver.fetch!()

    model_paths
    |> Enum.map(fn path ->
      with {:ok, {endpoint, config_file, model_file}} <- get_model_files(path),
           {:ok, config} <- File.read(config_file),
           {:ok, %ModelConfig{} = model_config} <- ModelConfig.parse!(config),
           {:ok, model_ref} <- Model.load(driver, model_file, model_config)
        {:ok, {endpoint, model_ref}}
      else
        # TODO: Match error
        _ ->
          Logger.error("Failed to load model #{endpoint}")
          {:error, :bad}
    end)
    |> Enum.filter(fn
      {:ok, _} -> true
      {:error, _} -> false
    end)
    |> Map.new()
  end

  defp get_model_files(path) do
    endpoint = Path.dirname(path) |> IO.inspect
    # TODO: Add verbose error for both failure paths
    config_file = Path.join(path, @config_file)
    model_file = Path.join(path, @model_file)

    cond do
      not File.exists?(model_file) ->
        Logger.error("Failed to load model #{endpoint}. Could not find model file")
        {:error, :missing_model_file}

      not File.exists?(config_file) ->
        Logger.error("Failed to load model #{endpoint}. Could not find config file")
        {:error, :missing_config_file}

      true ->
        {:ok, {endpoint, config_file, model_file}}
    end
  end

  def start_link(repo, opts \\ []) do
    GenServer.start_link(__MODULE__, repo, name: __MODULE__)
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
