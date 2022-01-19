defmodule TPUServe.Endpoint do
  use Plug.Router
  use Plug.ErrorHandler

  plug(Plug.Logger)
  plug(:match)

  plug(Plug.Parsers,
    parsers: [:json, Msgpax.PlugParser],
    json_decoder: Jason,
    pass: ["application/json", "application/msgpack"]
  )

  plug(:dispatch)

  get "/status" do
    send_resp(conn, 200, "Up")
  end

  post "/inference/v1/:endpoint" do
    inference_params = conn.body_params
    content_type = get_req_header(conn, "content-type")

    with {:ok, model} <- TPUServe.ModelManager.fetch_model(endpoint),
         {:ok, inference_result} <- TPUServe.InferenceHandler.predict(model, inference_params),
         {:ok, response_body} <- TPUServe.Protocol.encode_response(inference_result, content_type) do
      send_resp(conn, 200, response_body)
    else
      # TODO: Errors :)
      send_resp(conn, 404, "sorry")
    end
  end

  match _ do
    send_resp(conn, 404, "not found")
  end

  @impl Plug.ErrorHandler
  def handle_errors(conn, err) do
    IO.inspect(err.kind)
    IO.inspect(err.reason)
    IO.inspect(err.stack)
    send_resp(conn, conn.status, "Error")
  end
end
