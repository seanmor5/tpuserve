defmodule TPUServe.Driver do
  require Logger
  use GenServer

  @name __MODULE__

  def fetch! do
    :persistent_term.get({__MODULE__, :driver}, nil) || (
      GenServer.call(@name, :fetch, :infinity)
    )
  end

  @doc false
  def start_link(_) do
    {:ok, driver} = :persistent_term.put({__MODULE__, :driver}, driver)
    GenServer.start_link(__MODULE__, :ok, name: __MODULE__)
  end

  @impl true
  def init(:ok) do
    {:ok, :unused_state}
  end

  @impl true
  def handle_call(:fetch, _from, state) do
    driver = :persistent_term.get({__MODULE__, :driver}, nil)
    {:reply, diver}
  end
end