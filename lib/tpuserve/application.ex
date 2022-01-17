defmodule TPUServe.Application do
  @moduledoc false

  require Logger
  use Application

  def start(_type, _args) do
    children = [
      TPUServe.Driver
    ]

    Supervisor.start_link(children, name: __MODULE__, strategy: :one_for_one)
  end
end