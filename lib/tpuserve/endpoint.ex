defmodule TPUServe.Endpoint do
  use Plug.Router

  plug(Plug.Logger)
  plug(:match)
  plug(Plug.Parsers, parser: [:json], json_decoder: Jason)
  plug(:dispatch)

  get "/ping" do
    send_resp(conn, 200, "pong")
  end
end