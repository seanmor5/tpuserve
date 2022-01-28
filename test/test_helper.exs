defmodule TestUtils do

  @test_cache "test/.cache"

  @doc """
  Exports and tests function against the given test
  cases. Test cases is a map of case_name => List of
  input type/shape.
  """
  def export_and_test_model(name, fun, cases) do
    driver = TPUServe.Driver.fetch!()

    cases
    |> Enum.map(fn {case_name, inputs} ->
      case_path = Path.join([@test_cache, name, case_name])
      model_path = Path.join(case_path, "model.hlo.txt")

      inp_tensor_map = Map.new(inputs, fn {name, shape, type} ->
        {name, Nx.random_uniform(shape, type: type)}
      end)

      {:ok, loaded_model} =
        if File.exists?(model_path) do
          config = generate_model_config(case_name, inputs)
          TPUServe.Model.load(driver, model_path, config)
        else
          File.mkdir_p(case_path)
          config = generate_model_config(case_name, inputs)
          model_hlo = EXLA.export(fun, Map.values(inp_tensor_map))
          File.write!(model_path, model_hlo)
          TPUServe.Model.load(driver, model_path, config)
        end

      # Get expected
      expected = Nx.Defn.jit(fun, Map.values(inp_tensor_map))

      # Get actual
      inp_buffers = Enum.map(inp_tensor_map, fn {k, v} -> {k, Nx.to_binary(v)} end)
      {:ok, pred} = TPUServe.Model.predict(loaded_model, inp_buffers)

      # TODO: Handle Tuples
      actual = Nx.from_binary(pred, Nx.type(expected)) |> Nx.reshape(expected)
      assert_all_close!(actual, expected)
    end)
  end

  defp generate_model_config(case_name, inputs) do
    inp_map =
      inputs
      |> Enum.map(fn {name, shape, type} -> %{name: name, shape: shape, type: type} end)

    # TODO: Do we ever need outputs?
    %TPUServe.ModelConfig{name: case_name, inputs: inp_map, outputs: []}
  end

  def assert_all_close!(lhs, rhs) do
    close? = Nx.Defn.jit(fn left, right -> Nx.all_close(left, right) end, [lhs, rhs])

    unless close? == Nx.tensor(1, type: {:u, 8}) do
      raise "expected #{inspect(lhs)} to be within tolerance of #{inspect(rhs)}"
    end
  end
end

ExUnit.start()

Nx.Defn.global_default_options(compiler: EXLA)
