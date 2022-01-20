defmodule TPUServe.Driver do
  require Logger
  use GenServer
  alias __MODULE__, as: Driver

  defstruct :ref

  # TODO: Is driver thread safe?

  def start_link(_) do
    Logger.info("Starting TPUServe Driver")
    {:ok, driver_ref} = TPUServe.NIF.init_driver()
    driver = %Driver{ref: driver_ref}
    :persistent_term.put({__MODULE__, :driver}, driver)
    GenServer.start_link(__MODULE__, :ok, name: __MODULE__)
  end

  def fetch! do
    :persistent_term.get({__MODULE__, :driver}, nil) || GenServer.call(@name, :fetch, :infinity)
  end

  @impl true
  def init(:ok) do
    {:ok, :unused_state}
  end

  @impl true
  def handle_call(:fetch, _from, _state) do
    driver = :persistent_term.get({__MODULE__, :driver}, nil)
    {:reply, driver}
  end
end
