defmodule TPUServe.Model do
  @moduledoc """
  Model Abstraction.
  """

  def load(driver, file, config) do
    input_sizes = Enum.map(config.inputs, &get_tensor_spec_size/1)
    output_sizes = Enum.map(config.outputs, &get_tensor_spec_size/1)

    {:ok, model} = TPUServe.NIF.load_model(driver, file, input_sizes, output_sizes)
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
