defmodule TPUServe.EndpointTest do
  use ExUnit.Case
  use Plug.Test

  alias TPUServe.Endpoint

  import Nx.Defn

  describe "status" do
    test "returns 200" do
      response = conn(:get, "/v1/status") |> Endpoint.call()

      assert response.status == 200
      assert response.body == "Up"
    end
  end

  describe "inference" do
    Nx.Defn.default_options(compiler: EXLA)

    defn prog(a, b), do: Nx.dot(a, b) / 3.14159

    test "multi-input, names in order, msgpack" do
      a = Nx.random_uniform({8, 128})
      b = Nx.random_uniform({128, 32})
      expected = prog(a, b)

      a_enc = a |> Nx.to_binary |> Msgpax.Bin.new()
      b_enc = a |> Nx.to_binary |> Msgpax.Bin.new()
      enc = %{a: a_enc, b: b_enc} |> Msgpax.pack!()

      response =
        conn(:post, "/v1/inference/prog", enc))
        |> put_req_header("content-type", "application/msgpack")
        |> Endpoint.call()

      assert response.status == 200
      assert Nx.from_binary(response.body, {:f, 32}) |> Nx.reshape({8, 32}) == expected
    end

    # TODO: multi-input, names out of order
    # TODO: single input, single output
    # TODO: multi-output
    # TODO: Lots of bad requests
  end
end
