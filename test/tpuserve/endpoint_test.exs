defmodule TPUServe.EndpointTest do
  use ExUnit.Case
  use Plug.Test

  alias TPUServe.Endpoint

  import Nx.Defn

  describe "status" do
    test "returns 200" do
      response = conn(:get, "/v1/status") |> Endpoint.call([])

      assert response.status == 200
      assert response.resp_body == "Up"
    end
  end

  describe "inference" do
    Nx.Defn.default_options(compiler: EXLA)
    defn simple_add(a, b), do: a + b

    test "simple, multi-input, names in order, msgpack" do
      a = Nx.tensor(1, type: {:s, 32})
      b = Nx.tensor(2, type: {:s, 32})
      expected = simple_add(a, b)

      a_enc = a |> Nx.to_binary() |> Msgpax.Bin.new()
      b_enc = b |> Nx.to_binary() |> Msgpax.Bin.new()
      enc = %{a: a_enc, b: b_enc} |> Msgpax.pack!() |> IO.iodata_to_binary()

      response =
        conn(:post, "/v1/inference/simple_add", enc)
        |> put_req_header("content-type", "application/msgpack")
        |> Endpoint.call([])

      assert response.status == 200
      res = Msgpax.unpack!(response.resp_body)
      assert Nx.from_binary(res, {:s, 32}) == expected
    end

    defn prog(a, b), do: Nx.dot(a, b) / 3.14159

    test "multi-input, names in order, msgpack" do
      a = Nx.broadcast(1.0, {8, 128})
      b = Nx.broadcast(2.0, {128, 32})
      expected = prog(a, b)

      a_enc = a |> Nx.to_binary |> Msgpax.Bin.new()
      b_enc = a |> Nx.to_binary |> Msgpax.Bin.new()
      enc = %{a: a_enc, b: b_enc} |> Msgpax.pack!() |> IO.iodata_to_binary()

      response =
        conn(:post, "/v1/inference/prog", enc)
        |> put_req_header("content-type", "application/msgpack")
        |> Endpoint.call([])

      assert response.status == 200
      bin = response.resp_body
      res = Msgpax.unpack!(bin)
      assert Nx.from_binary(res, {:f, 32}) |> Nx.reshape({8, 32}) == expected
    end

    # TODO: multi-input, names out of order
    # TODO: single input, single output
    # TODO: multi-output
    # TODO: Lots of bad requests
  end
end
