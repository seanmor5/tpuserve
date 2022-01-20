defmodule TPUServe.Model do
  @moduledoc """
  Model Abstraction.
  """

  alias TPUServe.Driver
  alias __MODULE__, as: Model

  defstruct [:driver, :ref]

  @doc """
  Loads the model.
  """
  def load(%Driver{ref: driver}, file, config) do
    input_sizes = Enum.map(config.inputs, &get_tensor_spec_size/1)
    output_sizes = Enum.map(config.outputs, &get_tensor_spec_size/1)

    {:ok, model_ref} = TPUServe.NIF.load_model(driver, file, input_sizes, output_sizes)
    model = %Model{driver: driver, ref: model_ref}

    {:ok, model}
  end

  @doc """
  Performs inference on inputs.
  """
  def predict(%Model{ref: model_ref}, inputs) do
    # TODO: Inputs should map to correctly ordered
    # buffers according to config
    input_buffers =
      inputs
      |> Map.values()

    TPUServe.NIF.predict(model_ref, input_buffers)
  end

  # TODO: This should be elsewhere
  defp get_tensor_spec_size(tensor_spec) do
    %{shape: shape, type: {_, type_size}} = tensor_spec
    type_byte_size = div(type_size, 8)

    shape
    |> Enum.reduce(1, fn x, acc -> x * acc end)
    |> Kernel.*(type_byte_size)
  end
end
