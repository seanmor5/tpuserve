defmodule TPUServe.Model do
  @moduledoc """
  Model Abstraction.
  """

  alias TPUServe.{Driver, Error}
  alias __MODULE__, as: Model

  defstruct [:driver, :ref, :config]

  @doc """
  Loads the model.
  """
  def load(%Driver{ref: driver}, file, config) do
    {:ok, model_ref} = TPUServe.NIF.load_model(driver, file)
    model = %Model{driver: driver, ref: model_ref, config: config}

    {:ok, model}
  end

  @doc """
  Performs inference on inputs.
  """
  def predict(%Model{ref: model_ref, config: config, driver: driver_ref}, inputs) do
    input_buffers =
      config.inputs
      |> Enum.map(fn inp ->
        if Map.has_key?(inputs, inp.name) do
          inputs[inp.name]
        else
          msg = "#{inp.name} not in model inputs"
          {:error, Error.inference(msg)}
        end
      end)

    # TODO: Map result to output map here
    case TPUServe.NIF.predict(driver_ref, model_ref, input_buffers) do
      {:ok, out} ->
        {:ok, out}

      {:error, msg} ->
        Error.inference(msg)
    end
  end
end
