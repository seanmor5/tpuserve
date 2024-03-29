defmodule TPUServe.ModelConfigTest do
  use ExUnit.Case

  alias TPUServe.ModelConfig

  describe "parse!" do
    @resnet_config """
    {
      "name": "ResNet50",
      "inputs": [
        {
          "name": "image",
          "shape": [1, 224, 224, 3],
          "type": "F32"
        }
      ],
      "outputs": [
        {
          "name": "class",
          "shape": [1, 1000],
          "type": "F32"
        }
      ]
    }
    """

    test "resnet config" do
      assert config = %ModelConfig{} = ModelConfig.parse!(@resnet_config)
      assert [image] = config.inputs
      assert [class] = config.outputs
      assert image.name == "image"
      assert class.name == "class"
    end

    @duplicate_input_names """
    {
      "name": "ResNet50",
      "inputs": [
        {
          "name": "image",
          "shape": [1, 224, 224, 3],
          "type": "F32"
        },
        {
          "name": "image",
          "shape": [1, 223, 223, 3],
          "type": "F32"
        }
      ],
      "outputs": [
        {
          "name": "class",
          "shape": [1, 1000],
          "type": "F32"
        }
      ]
    }
    """

    test "raises on duplicate input names" do
      assert_raise ArgumentError, ~r/tensor spec/, fn ->
        ModelConfig.parse!(@duplicate_input_names)
      end
    end

    @duplicate_output_names """
    {
      "name": "ResNet50",
      "inputs": [
        {
          "name": "image",
          "shape": [1, 224, 224, 3],
          "type": "F32"
        }
      ],
      "outputs": [
        {
          "name": "class",
          "shape": [1, 1000],
          "type": "F32"
        },
        {
          "name": "class",
          "shape": [2, 1],
          "type": "F32"
        }
      ]
    }
    """

    test "raises on duplicate output names" do
      assert_raise ArgumentError, ~r/tensor spec/, fn ->
        ModelConfig.parse!(@duplicate_output_names)
      end
    end

    # TODO: Test bad shape
    # TODO: Test no inputs
    # TODO: Test no outputs
    # TODO: Test missing info
    # TODO: Test bad / unsupported type
  end
end
