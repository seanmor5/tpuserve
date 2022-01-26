defmodule TPUServe.MixProject do
  use Mix.Project

  def project do
    [
      app: :tpuserve,
      version: "0.1.0",
      elixir: "~> 1.12",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      compilers: [:elixir_make] ++ Mix.compilers()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger],
      mod: {TPUServe.Application, []}
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:plug_cowboy, "~> 2.0"},
      {:jason, "~> 1.2"},
      {:elixir_make, "~> 0.6", runtime: false},
      {:msgpax, "~> 2.3.0"},
      {:exla, "~> 0.1.0-dev", github: "elixir-nx/nx", sparse: "exla", only: [:test]},
      {:nx, "~> 0.1.0", only: [:test], override: true}
    ]
  end
end
