defmodule TPUServe.ModelManager do
  @moduledoc """
  Manages which models are exposed by the server.

  Stores TPUServeModel references in a map of
  `%{endpoint => model_ref}`. TPUServeModels are just
  loaded program handles and input/output buffer handles.
  """

  @model_extension ".hlo.txt"

  require Logger
  use GenServer

  def init(repo) do
    model_paths =
      repo
      |> Path.join("*" <> @model_extension)
      |> Path.wildcard()

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
    driver = TPUServe.Driver.fetch!()

    model_paths
    |> Enum.map(fn path -> {Path.basename(path, @model_extension), path} end)
    |> Map.new(fn {endpoint, path} ->
      {:ok, model_ref} = TPUServe.NIF.load_model(driver, path)
      {endpoint, model_ref}
    end)
  end

  def start_link(repo, opts \\ []) do
    GenServer.start_link(__MODULE__, repo, name: __MODULE__)
  end

  def handle_call({:fetch, model}, _from, state) do
    case state[model] do
      nil ->
        {:reply, {:error, :not_found}, state}

      model ->
        {:reply, {:ok, model}, state}
    end
  end
end
