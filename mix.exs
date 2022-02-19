defmodule TPUServe.MixProject do
  use Mix.Project

  def project do
    [
      app: :tpuserve,
      version: "0.1.0",
      releases: releases(),
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

  def releases do
    [
      tpuserve: [
        steps: [:assemble, &Burrito.wrap/1],
        burrito: [
          targets: [
            linux: [os: :linux, cpu: :x86_64]
          ]
        ]
      ]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:plug_cowboy, "~> 2.0"},
      {:jason, "~> 1.2"},
      {:elixir_make, "~> 0.6", runtime: false},
      {:msgpax, "~> 2.3.0"},
      {:burrito, github: "burrito-elixir/burrito"},
      {:exla, "~> 0.1.0-dev",
       github: "elixir-nx/nx", sparse: "exla", branch: "sm-exla-export", only: [:test]},
      {:nx, "~> 0.1.0-dev",
       github: "elixir-nx/nx", sparse: "nx", branch: "sm-exla-export", override: true}
    ]
  end
end
