defmodule TPUServe.ModelConfig do
  @moduledoc """
  Model configuration.
  """
  alias __MODULE__, as: ModelConfig

  @enforce_keys [:name, :inputs, :outputs]
  defstruct [:name, :inputs, :outputs]

  def parse!(config) do
    config
    |> Jason.decode!()
    |> normalize!()
    |> then(&struct(ModelConfig, &1))
  end

  # Get config into format accepted by model manager,
  # and do as much validation as possible so we don't
  # screw up the NIF
  defp normalize!(config) do
    %{"name" => name, "inputs" => inputs, "outputs" => outputs} = config

    name = normalize_name!(name)
    inputs =
      inputs
      |> Enum.map(&normalize_tensor_spec!/1)
      |> ensure_unique_names!()

    outputs =
      outputs
      |> Enum.map(&normalize_tensor_spec!/1)
      |> ensure_unique_names!()

    %{
      name: name,
      inputs: inputs,
      outputs: outputs
     }
  end

  # Name represents the key in the model manager and
  # as a consequence needs to be acceptable as a URL
  # endpoint - TODO
  defp normalize_name!(name) do
    name
  end

  # Tensor spec is a name, a shape, and a type. We need
  # to validate the shape to ensure all dimensions are
  # strictly positive and the type to ensure it is an
  # acceptable TPU type
  defp normalize_tensor_spec!(tensor_spec) do
    %{"name" => name, "shape" => shape, "type" => type} = tensor_spec

    # TODO: name
    shape = normalize_shape!(name, shape)
    type = normalize_type!(name, type)

    %{name: name, shape: shape, type: type}
  end

  defp normalize_shape!(name, shape) when is_list(shape) do
    valid_shape? =
      shape
      |> Enum.map(& &1 > 0)
      |> Enum.all?()

    unless valid_shape? do
      raise ArgumentError, "invalid shape for tensor spec #{inspect(name)}:" <>
                             " #{inspect(shape)}"
    end

    shape
  end

  defp normalize_shape!(name, shape) do
    # TODO: Custom Error
    raise ArgumentError, "inavlid shape for tensor spec #{inspect(name)}:" <>
                            " #{inspect(shape)}"
  end

  defp normalize_type!(name, type) when is_binary(type) do
    case type do
      "BF16" -> {:bf, 16}
      "F32" -> {:f, 32}
      "S32" -> {:s, 32}
    end
  end

  defp normalize_type!(name, type) do
    # TODO: Custom Error
    raise ArgumentError, "invalid type for tensor spec #{inspect(name)}:" <>
                            " #{inspect(type)}"
  end

  defp ensure_unique_names!(tensor_specs) do
    are_all_unique? =
      tensor_specs
      |> Enum.map(& &1.name)
      |> then(&Enum.count(&1) == Enum.count(Enum.uniq(&1)))

    unless are_all_unique? do
      raise ArgumentError, "tensor spec names must all be unique"
    end

    tensor_specs
  end
end
