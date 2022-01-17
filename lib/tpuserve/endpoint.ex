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
    inference_params = conn.body_params

    case TPUServe.ModelManager.fetch_model(endpoint) do
      {:ok, model} ->
        reply = TPUServe.InferenceHandler.predict(model, inference_params)
        send_resp(conn, 200, reply)

      _ ->
        send_resp(conn, 404, "not found")
    end
  end

  match _ do
    send_resp(conn, 404, "not found")
  end
end
