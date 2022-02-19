defmodule TPUServe.Error do
  defexception [:code, :message, :meta]

  def new(code, msg, meta) when is_binary(msg) do
    %__MODULE__{code: code, message: msg, meta: meta}
  end

  def not_found(msg, meta \\ %{}) do
    new(:not_found, msg, meta)
  end

  def inference(msg, meta \\ %{}) do
    new(:inference, msg, meta)
  end

  def internal(msg, meta \\ %{}) do
    new(:internal, msg, meta)
  end
end
