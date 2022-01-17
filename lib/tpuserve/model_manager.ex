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
      %{} ->
        Logger.warn("Manager did not successfully load any models")
        {:ok, %{}}

      models ->
        endpoints = Map.keys(models)
        Logger.info("Successfully loaded models for endpoints #{inspect(endpoints)}")
        {:ok, models}
    end
  end

  defp try_load_models(model_paths) do
    model_paths
    |> Enum.map(fn path -> {Path.basename(path, @model_extension), path} end)
    |> Map.new(fn {endpoint, path} ->
      {endpoint, path}
    end)
  end

  def start_link(repo, opts \\ []) do
    GenServe.start_link(__MODULE__, repo, opts)
  end
end