defmodule TPUServe.Endpoint do
  use Plug.Router
  use Plug.ErrorHandler

  alias TPUServe.{Model, ModelManager, Protocol}

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

    with {:ok, %Model{} = model} <- ModelManager.fetch(endpoint),
         {:ok, inference_result} <- Model.predict(model, inference_params),
         {:ok, response_body} <- Protocol.encode(inference_result, content_type) do
      send_resp(conn, 200, response_body)
    else
      _ ->
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
