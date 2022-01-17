defmodule TPUServe.Endpoint do
  use Plug.Router

  plug Plug.Logger
  plug :match
  plug Plug.Parsers, parsers: [:json],
                     json_decoder: Jason
  plug :dispatch

  get "/status" do
    send_resp(conn, 200, "Up")
  end

  post "/:endpoint" do
    IO.inspect conn.body_params
    case TPUServe.ModelManager.fetch_model(endpoint) do
      {:ok, model} ->
        send_resp(conn, 200, "Success")

      _ ->
        send_resp(conn, 404, "not found")
    end
  end

  match _ do
    send_resp(conn, 404, "not found")
  end
end
