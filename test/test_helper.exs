defmodule TestUtils do

  def assert_all_close!(lhs, rhs) do
    close? = Nx.Defn.jit(fn left, right -> Nx.all_close(left, right) end, [lhs, rhs])

    unless close? == Nx.tensor(1, type: {:u, 8}) do
      raise "expected #{inspect(lhs)} to be within tolerance of #{inspect(rhs)}"
    end
  end
end

ExUnit.start()

Nx.Defn.global_default_options(compiler: EXLA)
