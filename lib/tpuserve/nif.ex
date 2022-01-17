defmodule TPUServe.NIF do
  @moduledoc false
  @on_load :__on_load__

  def __on_load__ do
    path = :filename.join(:code.priv_dir(:tpuserve), 'libtpuserve')
    :erlang.load_nif(path, 0)
  end

  def init_driver, do: :erlang.nif_error(:undef)

  def load_model(_driver, _model_path), do: :erlang.nif_error(:undef)
end
