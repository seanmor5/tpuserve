defmodule TpuserveTest do
  use ExUnit.Case
  doctest Tpuserve

  test "greets the world" do
    assert Tpuserve.hello() == :world
  end
end
