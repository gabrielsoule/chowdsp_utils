#pragma once

namespace chowdsp
{
template <int degree, typename T = float>
class SoftClipper
{
    static_assert (degree % 2 == 1 && degree > 2, "Degree must be an odd integer, larger than 2!");

public:
    SoftClipper() = default;

    void prepare (const juce::dsp::ProcessSpec& spec)
    {
        exponentData = std::vector<T> (spec.maximumBlockSize, (T) 0);
    }

    static inline T processSample (T x) noexcept
    {
        x = juce::jlimit ((T) -1, (T) 1, x * normFactor);
        x = x - Power::ipow<degree> (x) * oneOverDeg;
        return x * invNormFactor;
    }

    template <typename ProcessContext>
    void process (const ProcessContext& context) noexcept
    {
        const auto& inputBlock = context.getInputBlock();
        auto& outputBlock = context.getOutputBlock();

        const auto numSamples = (int) inputBlock.getNumSamples();
        const auto numChannels = inputBlock.getNumChannels();
        jassert (outputBlock.getNumChannels() == numChannels);

        for (size_t ch = 0; ch < numChannels; ++ch)
        {
            T* channelData;
            if (context.usesSeparateInputAndOutputBlocks())
            {
                channelData = outputBlock.getChannelPointer (ch);
                juce::FloatVectorOperations::copy (channelData, inputBlock.getChannelPointer (ch), numSamples);
            }
            else
            {
                channelData = outputBlock.getChannelPointer (ch);
            }

            if (context.isBypassed)
                continue;

            juce::FloatVectorOperations::multiply (channelData, normFactor, numSamples);
            juce::FloatVectorOperations::clip (channelData, channelData, (T) -1, (T) 1, numSamples);

            FloatVectorOperations::integerPower (exponentData.data(), channelData, degree, numSamples);
            juce::FloatVectorOperations::multiply (exponentData.data(), oneOverDeg, numSamples);
            juce::FloatVectorOperations::subtract (channelData, exponentData.data(), numSamples);

            juce::FloatVectorOperations::multiply (channelData, invNormFactor, numSamples);
        }
    }

private:
    static constexpr auto oneOverDeg = (T) 1 / (T) degree;
    static constexpr auto normFactor = T (degree - 1) / (T) degree;
    static constexpr auto invNormFactor = (T) 1 / normFactor;

    std::vector<T> exponentData;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (SoftClipper)
};

} // namespace chowdsp