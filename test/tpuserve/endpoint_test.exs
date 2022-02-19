defmodule TPUServe.EndpointTest do
  use ExUnit.Case
  use Plug.Test

  alias TPUServe.Endpoint

  import Nx.Defn
  import TestUtils

  describe "status" do
    test "returns 200" do
      response = conn(:get, "/v1/status") |> Endpoint.call([])

      assert response.status == 200
      assert response.resp_body == "Up"
    end
  end
end
