defmodule TPUServe.ModelTest do
  use ExUnit.Case, async: true

  import TestUtils
  import Nx.Defn

  @unsigned_types [{:u, 8}, {:u, 16}, {:u, 32}, {:u, 64}]
  @signed_types [{:s, 8}, {:s, 16}, {:s, 32}, {:s, 64}]
  @float_types [{:bf, 16}, {:f, 16}, {:f, 32}, {:f, 64}]
  @all_types (@unsigned_types ++ @signed_types ++ @float_types)

  describe "simple nx node tests" do
    test "no-input, constant scalar across types" do
      for type <- @all_types do
        fun = fn -> Nx.tensor(1, type: type) end
        cases = %{"type_#{Nx.Type.to_string(type)}" => []}
        export_and_test_model("constant", fun, cases)
      end
    end

    defn identity(x), do: x

    test "identity, across types, shapes" do
      shapes = rank_up_to(5)
      cases =
        for shape <- shapes,
            type <- @all_types, into: %{} do
          rank = Nx.rank(shape)
          {"rank_#{rank}_type_#{Nx.Type.to_string(type)}", [{"a", shape, type}]}
        end

      export_and_test_model("identity", &identity/1, cases)
    end
  end

  describe "element-wise node tests" do
    element_wise_ops = [:abs, :ceil, :floor, :negate, :sign]

    for op <- element_wise_ops do
      test "#{op}, across types, shapes" do
        shapes = rank_up_to(5)
        cases =
          for shape <- shapes,
              type <- @all_types, into: %{} do
            rank = Nx.rank(shape)
            {"rank_#{rank}_type_#{Nx.Type.to_string(type)}", [{"a", shape, type}]}
          end

        export_and_test_model(Atom.to_string(unquote(op)), &apply(Nx, unquote(op), [&1]), cases)
      end
    end
  end

  describe "aggregate tests" do
    multi_axis_aggregate_ops = [:mean, :product, :reduce_max, :reduce_min, :sum]

    for op <- multi_axis_aggregate_ops do
      test "#{op}, all axes, across types, shapes" do
        shapes = rank_up_to(5)
        cases =
          for shape <- shapes,
              type <- @all_types, into: %{} do
            rank = Nx.rank(shape)
            {"rank_#{rank}_type_#{Nx.Type.to_string(type)}", [{"a", shape, type}]}
          end

        export_and_test_model(Atom.to_string(unquote(op)), &apply(Nx, unquote(op), [&1]), cases)
      end

      # test "#{op}, last axis, across types, shapes" do
      #   [_ | shapes] = rank_up_to(5)
      #   cases =
      #     for shape <- shapes,
      #         type <- @all_types, into: %{} do
      #       rank = Nx.rank(shape)
      #       {"rank_#{rank}_type_#{Nx.Type.to_string(type)}", [{"a", shape, type}]}
      #     end

      #   export_and_test_model(Atom.to_string(unquote(op)), &apply(Nx, unquote(op), [&1, [axes: [-1]]]), cases)
      # end

      # test "#{op}, first and last axis, across types, shapes" do
      #   [_, _ | shapes] = rank_up_to(5)
      #   cases =
      #     for shape <- shapes,
      #         type <- @all_types, into: %{} do
      #       rank = Nx.rank(shape)
      #       {"rank_#{rank}_type_#{Nx.Type.to_string(type)}", [{"a", shape, type}]}
      #     end

      #   export_and_test_model(Atom.to_string(unquote(op)), &apply(Nx, unquote(op), [&1, [axes: [0, -1]]]), cases)
      # end
    end
  end
end