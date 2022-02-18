defmodule TestUtils do
  @test_cache "test/.cache"

  @doc """
  Exports and tests function against the given test
  cases. Test cases is a map of case_name => List of
  input type/shape.
  """

  def default_tensor_generator(shape, type) do
    case type do
      {:s, _} ->
        Nx.random_uniform(shape, -5, 5, type: type)

      {:u, _} ->
        Nx.random_uniform(shape, 0, 10, type: type)

      _ ->
        Nx.random_uniform(shape, type: type)
    end
  end

  def rank_up_to(n) do
    shapes =
      for i <- 1..n do
        for _ <- 0..(i - 1) do
          :rand.uniform(10)
        end
        |> List.to_tuple()
      end

    [{} | shapes]
  end

  def broadcastable(shape) do
    case shape do
      {} ->
        {}

      shape when is_tuple(shape) ->
        rank = Nx.rank(shape)
        random_rank = :rand.uniform(rank) - 1
        put_elem(shape, random_rank, 1)
    end
  end

  def export_and_test_model(name, fun, cases, generator \\ &default_tensor_generator/2) do
    driver = TPUServe.Driver.fetch!()

    cases
    |> Enum.map(fn {case_name, inputs} ->
      try do
        case_path = Path.join([@test_cache, name, case_name])
        model_path = Path.join(case_path, "model.hlo.txt")

        inp_tensor_map =
          Map.new(inputs, fn {name, shape, type} ->
            {name, generator.(shape, type)}
          end)

        File.mkdir_p(case_path)
        config = generate_model_config(case_name, inputs)
        model_hlo = EXLA.export(fun, Map.values(inp_tensor_map))
        File.write!(model_path, model_hlo)

        {:ok, loaded_model} = TPUServe.Model.load(driver, model_path, config)

        # Get expected
        expected = Nx.Defn.jit(fun, Map.values(inp_tensor_map), precision: :highest)

        # Get actual
        inp_buffers = Map.new(inp_tensor_map, fn {k, v} -> {k, Nx.to_binary(v)} end)
        {:ok, pred} = TPUServe.Model.predict(loaded_model, inp_buffers)

        # TODO: Handle Tuples
        actual = Nx.from_binary(pred, Nx.type(expected)) |> Nx.reshape(expected)
        assert_all_close!(actual, expected)
      rescue
        e ->
          reraise ArgumentError.exception(
                    message: "Test #{name}/#{case_name} failed: #{Exception.message(e)}"
                  ),
                  __STACKTRACE__
      end
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
    # TPUs are not very precise, but that's okay :D
    atol = 1.0

    close? = Nx.Defn.jit(fn left, right -> Nx.all_close(left, right, atol: atol) end, [lhs, rhs])

    unless close? == Nx.tensor(1, type: {:u, 8}) do
      raise "expected #{inspect(lhs)} to be within tolerance of #{inspect(rhs)}"
    end
  end
end

ExUnit.start()

Nx.Defn.global_default_options(compiler: EXLA)
