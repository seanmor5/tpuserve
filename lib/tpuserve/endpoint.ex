defmodule TPUServe.Endpoint do
  use Plug.Router
  use Plug.ErrorHandler

  alias TPUServe.{Error, Model, ModelManager, Protocol}

  plug(Plug.Logger)
  plug(:match)

  plug(Plug.Parsers,
    parsers: [:json, Msgpax.PlugParser],
    json_decoder: Jason,
    pass: ["application/json", "application/msgpack"]
  )

  plug(:dispatch)

  get "/v1/status" do
    send_resp(conn, 200, "Up")
  end

  get "/v1/list_models" do
    with {:ok, models} <- ModelManager.list(),
         {:ok, response_body} <- Protocol.encode_models(models) do
      send_resp(conn, 200, response_body)
    else
      {:error, e} ->
        send_error(conn, e)
    end
  end

  post "/v1/inference/:endpoint" do
    inference_params = conn.body_params
    content_type = get_req_header(conn, "content-type")

    with {:ok, %Model{} = model} <- ModelManager.fetch(endpoint),
         {:ok, inference_result} <- Model.predict(model, inference_params),
         {:ok, response_body} <- Protocol.encode(inference_result, content_type) do
      send_resp(conn, 200, response_body)
    else
      {:error, e} ->
        send_error(conn, e)
    end
  end

  match _ do
    send_error(conn, Error.not_found("Resource not found"))
  end

  @impl Plug.ErrorHandler
  def handle_errors(conn, _err) do
    send_error(conn, Error.internal("Internal error"))
  end

  defp send_error(conn, %Error{code: :not_found, message: msg}) do
    send_resp(conn, 404, msg)
  end

  defp send_error(conn, %Error{code: :inference, message: msg}) do
    send_resp(conn, 400, msg)
  end

  defp send_error(conn, %Error{code: :internal, message: msg}) do
    send_resp(conn, 500, msg)
  end
end
