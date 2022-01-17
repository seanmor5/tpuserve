defmodule TPUServe.Application do
  @moduledoc false

  require Logger
  use Application

  def start(_type, _args) do
    endpoint = Plug.Cowboy.child_spec(
      scheme: :http,
      plug: TPUServe.Endpoint,
      options: [port: 4000] # TODO: option
    )

    children = [
      TPUServe.Driver,
      {TPUServe.ModelManager, ["models"]},
      endpoint
    ]

    Supervisor.start_link(children, name: __MODULE__, strategy: :one_for_one)
  end
end
