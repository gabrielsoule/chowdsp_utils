#include "chowdsp_StateVariableFilter.h"

namespace chowdsp
{
template <typename SampleType, StateVariableFilterType type, size_t maxChannelCount, bool unityGain>
StateVariableFilter<SampleType, type, maxChannelCount, unityGain>::StateVariableFilter()
{
    if constexpr (type == FilterType::MultiMode)
    {
        setMode(0);
    }
    setCutoffFrequency (static_cast<NumericType> (1000.0));
    setQValue (static_cast<NumericType> (1.0 / juce::MathConstants<double>::sqrt2));
    setGain (static_cast<NumericType> (1.0));
}

template <typename SampleType, StateVariableFilterType type, size_t maxChannelCount, bool unityGain>
template <bool shouldUpdate>
void StateVariableFilter<SampleType, type, maxChannelCount, unityGain>::setCutoffFrequency (SampleType newCutoffFrequencyHz)
{
    jassert (SIMDUtils::all (newCutoffFrequencyHz >= static_cast<NumericType> (0)));
    jassert (SIMDUtils::all (newCutoffFrequencyHz < static_cast<NumericType> (sampleRate * 0.5)));

    cutoffFrequency = newCutoffFrequencyHz;
    const auto w = juce::MathConstants<NumericType>::pi * cutoffFrequency / (NumericType) sampleRate;

    CHOWDSP_USING_XSIMD_STD (tan);
    g0 = tan (w);

    if constexpr (shouldUpdate)
        update();
}

template <typename SampleType, StateVariableFilterType type, size_t maxChannelCount, bool unityGain>
template <bool shouldUpdate>
void StateVariableFilter<SampleType, type, maxChannelCount, unityGain>::setQValue (SampleType newResonance)
{
    jassert (SIMDUtils::all (newResonance > static_cast<NumericType> (0)));

    resonance = newResonance;
    k0 = (NumericType) 1.0 / resonance;
    k0A = k0 * A;

    if constexpr (shouldUpdate)
        update();
}

template <typename SampleType, StateVariableFilterType type, size_t maxChannelCount, bool unityGain>
template <bool shouldUpdate>
void StateVariableFilter<SampleType, type, maxChannelCount, unityGain>::setGain (SampleType newGainLinear)
{
    jassert (SIMDUtils::all (newGainLinear > static_cast<NumericType> (0)));

    gain = newGainLinear;

    CHOWDSP_USING_XSIMD_STD (sqrt);
    A = sqrt (gain);
    sqrtA = sqrt (A);
    Asq = A * A;
    k0A = k0 * A;

    if constexpr (shouldUpdate)
        update();
}

template <typename SampleType, StateVariableFilterType type, size_t maxChannelCount, bool unityGain>
template <bool shouldUpdate>
void StateVariableFilter<SampleType, type, maxChannelCount, unityGain>::setGainDecibels (SampleType newGainDecibels)
{
    setGain<shouldUpdate> (SIMDUtils::decibelsToGain (newGainDecibels));
}

template <typename SampleType, StateVariableFilterType type, size_t maxChannelCount, bool unityGain>
template <StateVariableFilterType M>
std::enable_if_t<M == StateVariableFilterType::MultiMode, void>
    StateVariableFilter<SampleType, type, maxChannelCount, unityGain>::setMode (NumericType mode)
{
    this->mode = mode;
    lowpassMult = (NumericType) 1 - (NumericType) 2 * juce::jmin ((NumericType) 0.5, mode);
    bandpassMult = (NumericType) 1 - std::abs ((NumericType) 2 * (mode - (NumericType) 0.5));
    highpassMult = (NumericType) 2 * juce::jmax ((NumericType) 0.5, mode) - (NumericType) 1;

    if constexpr(!unityGain)
    {
        // use sin3db power law for mixing
        lowpassMult = std::sin (juce::MathConstants<NumericType>::halfPi * lowpassMult);
        bandpassMult = std::sin (juce::MathConstants<NumericType>::halfPi * bandpassMult);
        highpassMult = std::sin (juce::MathConstants<NumericType>::halfPi * highpassMult);

        // the BPF is a little bit quieter by design, so let's compensate here for a smooth transition
        bandpassMult *= juce::MathConstants<NumericType>::sqrt2;
    }

}

template <typename SampleType, StateVariableFilterType type, size_t maxChannelCount, bool unityGain>
template <StateVariableFilterType M>
std::enable_if_t<M == StateVariableFilterType::MultiMode, void>
    StateVariableFilter<SampleType, type, maxChannelCount, unityGain>::updateParameters (SampleType newFrequency, SampleType newResonance, NumericType newMode)
{
    bool flag = false;
    if (newFrequency != cutoffFrequency)
    {
        setCutoffFrequency<false> (newFrequency);
        flag = true;
    }
    if (newResonance != resonance)
    {
        setQValue<false> (newResonance);
        flag = true;
    }
    if (newMode != mode)
    {
        setMode (newMode);
        flag = true;
    }
    if(flag)
    {
        update();
    }

}

template <typename SampleType, StateVariableFilterType type, size_t maxChannelCount, bool unityGain>
void StateVariableFilter<SampleType, type, maxChannelCount, unityGain>::prepare (const juce::dsp::ProcessSpec& spec)
{
    jassert (spec.sampleRate > 0);
    jassert (spec.numChannels > 0);

    sampleRate = spec.sampleRate;

    if constexpr (maxChannelCount == dynamicChannelCount)
    {
        ic1eq.resize (spec.numChannels);
        ic2eq.resize (spec.numChannels);
    }
    else
    {
        jassert (spec.numChannels <= maxChannelCount);
    }

    reset();

    setCutoffFrequency (cutoffFrequency);
}

template <typename SampleType, StateVariableFilterType type, size_t maxChannelCount, bool unityGain>
void StateVariableFilter<SampleType, type, maxChannelCount, unityGain>::reset()
{
    for (auto v : { &ic1eq, &ic2eq })
        std::fill (v->begin(), v->end(), static_cast<SampleType> (0));
}

template <typename SampleType, StateVariableFilterType type, size_t maxChannelCount, bool unityGain>
void StateVariableFilter<SampleType, type, maxChannelCount, unityGain>::snapToZero() noexcept // NOSONAR (cannot be const)
{
#if JUCE_SNAP_TO_ZERO
    for (auto v : { &ic1eq, &ic2eq })
        for (auto& element : *v)
            juce::dsp::util::snapToZero (element);
#endif
}

template <typename SampleType, StateVariableFilterType type, size_t maxChannelCount, bool unityGain>
void StateVariableFilter<SampleType, type, maxChannelCount, unityGain>::update()
{
    SampleType g, k;
    if constexpr (type == FilterType::Bell)
    {
        g = g0;
        k = k0 / A;
    }
    else if constexpr (type == FilterType::LowShelf)
    {
        g = g0 / sqrtA;
        k = k0;
    }
    else if constexpr (type == FilterType::HighShelf)
    {
        g = g0 * sqrtA;
        k = k0;
    }
    else
    {
        g = g0;
        k = k0;
    }

    const auto gk = g + k;
    a1 = (NumericType) 1.0 / ((NumericType) 1.0 + g * gk);
    a2 = g * a1;
    a3 = g * a2;
    ak = gk * a1;

    if constexpr(unityGain)
    {
        oneOverPeakGain = 1.0 / getPeakGain();
    }
}
} // namespace chowdsp
