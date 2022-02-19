defmodule TPUServe.ModelTest do
  use ExUnit.Case

  import TestUtils
  import Nx.Defn

  @unsigned_types [{:u, 8}, {:u, 16}, {:u, 32}]
  @signed_types [{:s, 8}, {:s, 16}, {:s, 32}]
  @float_types [{:bf, 16}, {:f, 16}, {:f, 32}]

  @integer_types @unsigned_types ++ @signed_types
  @all_types @unsigned_types ++ @signed_types ++ @float_types

  describe "simple nx node tests" do
    test "no-input, constant scalar across types" do
      for type <- @all_types do
        fun = fn -> Nx.tensor(1, type: type) end
        cases = %{"type_#{Nx.Type.to_string(type)}" => []}
        export_and_test_model("constant", fun, cases)
      end
    end

    defn(identity(x), do: x)

    test "identity, across types, shapes" do
      shapes = rank_up_to(2)

      cases =
        for shape <- shapes, type <- @all_types, into: %{} do
          rank = Nx.rank(shape)
          {"rank_#{rank}_type_#{Nx.Type.to_string(type)}", [{"a", shape, type}]}
        end

      export_and_test_model("identity", &identity/1, cases)
    end

    test "tuple outputs" do
      fun = fn x, y, z -> {x, y, x, z, y, x} end

      shapes = rank_up_to(2) |> Enum.shuffle() |> Enum.chunk_every(3, 1, :discard)
      mixed_types = Enum.shuffle(@all_types) |> Enum.chunk_every(3, 2, :discard)

      cases =
        for [s1, s2, s3] <- shapes, [t1, t2, t3] <- mixed_types, into: %{} do
          name =
            "a_type_#{Nx.Type.to_string(t1)}_b_type_#{Nx.Type.to_string(t2)}_c_type_#{Nx.Type.to_string(t3)}"

          {name, [{"a", s1, t1}, {"b", s2, t2}, {"c", s3, t3}]}
        end

      export_and_test_model("tuples", fun, cases)
    end
  end

  describe "element-wise node tests" do
    element_wise_ops = [
      :abs,
      :acos,
      :acosh,
      :asin,
      :asinh,
      :atan,
      :atanh,
      :cbrt,
      :ceil,
      :cos,
      :cosh,
      :erf,
      :erf_inv,
      :erfc,
      :exp,
      :expm1,
      :floor,
      :log,
      :log1p,
      :negate,
      :rsqrt,
      :sign,
      :sin,
      :sinh,
      :sqrt,
      :tan,
      :tanh
    ]

    for op <- element_wise_ops do
      test "#{op}, across types, shapes" do
        shapes = rank_up_to(2)

        cases =
          for shape <- shapes, type <- @all_types, into: %{} do
            rank = Nx.rank(shape)
            {"rank_#{rank}_type_#{Nx.Type.to_string(type)}", [{"a", shape, type}]}
          end

        export_and_test_model(Atom.to_string(unquote(op)), &apply(Nx, unquote(op), [&1]), cases)
      end
    end
  end

  describe "element-wise bitwise node tests" do
    element_wise_bitwise_ops = [:count_leading_zeros, :population_count, :bitwise_not]

    for op <- element_wise_bitwise_ops do
      test "#{op}, across types, shapes" do
        shapes = rank_up_to(2)

        cases =
          for shape <- shapes, type <- @integer_types, into: %{} do
            rank = Nx.rank(shape)
            {"rank_#{rank}_type_#{Nx.Type.to_string(type)}", [{"a", shape, type}]}
          end

        export_and_test_model(Atom.to_string(unquote(op)), &apply(Nx, unquote(op), [&1]), cases)
      end
    end
  end

  describe "element-wise binary node tests" do
    element_wise_binary_ops = [
      :add,
      :atan2,
      :divide,
      :equal,
      :greater,
      :greater_equal,
      :less,
      :less_equal,
      :max,
      :min,
      :multiply,
      :not_equal,
      :power,
      :remainder,
      :subtract
    ]

    for op <- element_wise_binary_ops do
      test "#{op}, across types, shapes" do
        shapes = rank_up_to(2)

        cases =
          for shape <- shapes, type <- @all_types, into: %{} do
            rank = Nx.rank(shape)

            {"rank_#{rank}_type_#{Nx.Type.to_string(type)}",
             [{"a", shape, type}, {"b", shape, type}]}
          end

        export_and_test_model(
          Atom.to_string(unquote(op)),
          &apply(Nx, unquote(op), [&1, &2]),
          cases
        )
      end

      test "#{op}, across types, shapes, broadcasting" do
        shapes = rank_up_to(2) |> Enum.map(fn x -> {x, broadcastable(x)} end)

        cases =
          for {shape, bshape} <- shapes, type <- @all_types, into: %{} do
            rank = Nx.rank(shape)

            {"rank_#{rank}_type_#{Nx.Type.to_string(type)}",
             [{"a", shape, type}, {"b", bshape, type}]}
          end

        export_and_test_model(
          Atom.to_string(unquote(op)),
          &apply(Nx, unquote(op), [&1, &2]),
          cases
        )
      end

      test "#{op}, across types, shapes, mixed_types" do
        shapes = rank_up_to(2)
        mixed_types = Enum.shuffle(@all_types) |> Enum.chunk_every(2, 2, :discard)

        cases =
          for shape <- shapes, [t1, t2] <- mixed_types, into: %{} do
            rank = Nx.rank(shape)
            name = "rank_#{rank}_a_type_#{Nx.Type.to_string(t1)}_b_type_#{Nx.Type.to_string(t2)}"
            {name, [{"a", shape, t1}, {"b", shape, t2}]}
          end

        export_and_test_model(
          Atom.to_string(unquote(op)),
          &apply(Nx, unquote(op), [&1, &2]),
          cases
        )
      end
    end
  end

  describe "element-wise binary node tests integer domains" do
    element_wise_binary_ops = [
      :bitwise_and,
      :bitwise_or,
      :bitwise_xor,
      :left_shift,
      :quotient,
      :right_shift
    ]

    for op <- element_wise_binary_ops do
      test "#{op}, across types, shapes" do
        shapes = rank_up_to(2)

        cases =
          for shape <- shapes, type <- @integer_types, into: %{} do
            rank = Nx.rank(shape)

            {"rank_#{rank}_type_#{Nx.Type.to_string(type)}",
             [{"a", shape, type}, {"b", shape, type}]}
          end

        export_and_test_model(
          Atom.to_string(unquote(op)),
          &apply(Nx, unquote(op), [&1, &2]),
          cases
        )
      end

      test "#{op}, across types, shapes, broadcasting" do
        shapes = rank_up_to(2) |> Enum.map(fn x -> {x, broadcastable(x)} end)

        cases =
          for {shape, bshape} <- shapes, type <- @integer_types, into: %{} do
            rank = Nx.rank(shape)

            {"rank_#{rank}_type_#{Nx.Type.to_string(type)}",
             [{"a", shape, type}, {"b", bshape, type}]}
          end

        export_and_test_model(
          Atom.to_string(unquote(op)),
          &apply(Nx, unquote(op), [&1, &2]),
          cases
        )
      end

      test "#{op}, across types, shapes, mixed_types" do
        shapes = rank_up_to(2)
        mixed_types = Enum.shuffle(@integer_types) |> Enum.chunk_every(2, 2, :discard)

        cases =
          for shape <- shapes, [t1, t2] <- mixed_types, into: %{} do
            rank = Nx.rank(shape)
            name = "rank_#{rank}_a_type_#{Nx.Type.to_string(t1)}_b_type_#{Nx.Type.to_string(t2)}"
            {name, [{"a", shape, t1}, {"b", shape, t2}]}
          end

        export_and_test_model(
          Atom.to_string(unquote(op)),
          &apply(Nx, unquote(op), [&1, &2]),
          cases
        )
      end
    end
  end

  describe "aggregate tests" do
    multi_axis_aggregate_ops = [:mean, :product, :reduce_max, :reduce_min, :sum]

    for op <- multi_axis_aggregate_ops do
      test "#{op}, all axes, across types, shapes" do
        shapes = rank_up_to(2)

        cases =
          for shape <- shapes, type <- @all_types, into: %{} do
            rank = Nx.rank(shape)
            {"rank_#{rank}_type_#{Nx.Type.to_string(type)}", [{"a", shape, type}]}
          end

        export_and_test_model(Atom.to_string(unquote(op)), &apply(Nx, unquote(op), [&1]), cases)
      end

      test "#{op}, last axis, across types, shapes" do
        [_ | shapes] = rank_up_to(2)

        cases =
          for shape <- shapes, type <- @all_types, into: %{} do
            rank = Nx.rank(shape)
            {"rank_#{rank}_type_#{Nx.Type.to_string(type)}", [{"a", shape, type}]}
          end

        export_and_test_model(
          Atom.to_string(unquote(op)),
          &apply(Nx, unquote(op), [&1, [axes: [-1]]]),
          cases
        )
      end

      test "#{op}, first and last axis, across types, shapes" do
        [_, _ | shapes] = rank_up_to(2)

        cases =
          for shape <- shapes, type <- @all_types, into: %{} do
            rank = Nx.rank(shape)
            {"rank_#{rank}_type_#{Nx.Type.to_string(type)}", [{"a", shape, type}]}
          end

        export_and_test_model(
          Atom.to_string(unquote(op)),
          &apply(Nx, unquote(op), [&1, [axes: [0, -1]]]),
          cases
        )
      end
    end
  end
end
