defmodule TPUServe.ModelTest do
  use ExUnit.Case

  import TestUtils
  import Nx.Defn

  @unsigned_types [{:u, 8}, {:u, 16}, {:u, 32}, {:u, 64}]
  @signed_types [{:s, 8}, {:s, 16}, {:s, 32}, {:s, 64}]
  @float_types [{:bf, 16}, {:f, 16}, {:f, 32}, {:f, 64}]
  @all_types (@unsigned_types ++ @signed_types ++ @float_types)

  describe "simple node tests" do
    test "no-input, constant across types" do
      for type <- @all_types do
        fun = fn -> Nx.tensor(1, type: type) end
        cases = %{"type_#{Nx.Type.to_string(type)}" => []}
        export_and_test_model("constant", fun, cases)
      end
    end
  end
end