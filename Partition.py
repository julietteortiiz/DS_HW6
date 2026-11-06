"""HW06 entropy and feature selection, adapted from Sara Mathieson's code."""
from __future__ import annotations
from collections import Counter
import math
from typing import Union

# Numeric values occur only during the provided ARFF preprocessing step.
FeatureValue = Union[str, float]
FeatureValues = dict[str, FeatureValue]
FeatureDomains = dict[str, list[str]]


class Example:
    def __init__(self, features: FeatureValues, label: int) -> None:
        """Provided data object. Supplied ARFF labels are -1 and +1."""
        self.features: FeatureValues = features
        self.label: int = label


class Partition:
    """A dataset and ordered feature domains; calculations must not mutate it.

    Use empirical frequencies (no smoothing), base-2 logs, and zero for
    zero-probability entropy terms. Methods requiring a distribution raise
    ValueError for an empty partition. Unknown feature names raise KeyError.
    Selection methods return the first feature in F when scores tie exactly;
    they raise ValueError if F is empty. No method prints or prompts for input.
    """
    def __init__(self, data: list[Example], F: FeatureDomains) -> None:
        """Provided constructor; keep the existing data/F/n interface."""
        self.data: list[Example] = data
        self.F: FeatureDomains = F
        self.n: int = len(data)

    def _require_data(self) -> None:
        """Provided validation helper for distribution calculations."""
        if not self.n:
            raise ValueError("A nonempty partition is required")

    def _prob(self, c: int) -> float:
        """Return P(Y=c). A class absent from a nonempty dataset gets zero."""
        #print("TESTING PROB " + str(c))
        count = 0
        for example in self.data:
            if example.label == c:
                count += 1
        #print("Prob for " + str(c) + ": " + str(count / self.n))
        if self.n == 0:
            return 0

        return count / self.n  

    def _probX(self, f: str, val: str) -> float:
        """Return P(X_f=val); an unobserved value contributes zero."""
        count = 0
        for example in self.data:
            if example.features.get(f) == val:
                count += 1
        return count / self.n

    def _cond_prob(self, f: str, val: str, c: int) -> float:
        """Return empirical P(Y=c | X_f=val), or 0 for an empty group.

        The empty-group convention supports zero-weight terms, not a claim
        that a conditional distribution on an impossible event is defined.
        """
        tot = 0
        count = 0
        #print("TESTING COND PROB ")
        for example in self.data:
            if example.features.get(f) == val:
                tot += 1
                if example.label == c:
                    count += 1
        #print("Prob for feature: " + f + "and val: " + val + "and label: " + str(c) +  "=" + str(count / tot))
        if tot == 0:
            return 0
        return count / tot

   
    def entropy(self) -> float:
        """Return H(Y) in bits; ignore zero-count classes."""
        labels = [-1, 1]
        h = []
        for label in labels:
            p = self._prob(label)
            ig = 0
            if p != 0:
                ig = -math.log2(p)
            h.append(p*ig)
        #print("TESTING ENTROPY IS: " + str(sum(h)))
        return sum(h)     

    def _cond_entropy(self, feature: str, val: str) -> float:
        """Return H(Y | X_feature=val), or 0 for an unobserved value."""
        hyx = 0
        labels = [-1, 1]
        for label in labels:
            cp = self._cond_prob(feature, val, label)
            ig = 0
            if cp != 0:
                ig = -math.log2(cp)
            hyx += cp * ig
        #print("TESTING _COND ENTROPY")
        #print("Entropy for feature: " + feature + "and val: " + val + "= " + str(hyx))
        return hyx

    def conditional_entropy(self, feature: str) -> float:
        """Return H(Y | X_feature), weighted by group sizes, not a simple mean."""
        hyx = 0
        for val in self.F.get(feature):
            px = self._probX(feature, val)
            cond_p = self._cond_entropy(feature, val)
            hyx += px * cond_p
        #print("TESTING COND ENTROPY WHOLE FEATURE")
        #print("Entropy for feature: " + feature + "= " + str(hyx))
        return hyx     

    def information_gain(self, feature: str) -> float:
        """Return H(Y) - H(Y | X_feature), in bits."""
        #print("TESTING INFORMATION GAIN")
        #print( "IG: " + str(self.entropy() - self.conditional_entropy(feature)))
        return self.entropy() - self.conditional_entropy(feature)

    def classification_accuracy(self, feature: str) -> float:
        """Return training accuracy when each feature-value group predicts its majority.
        A label tie does not change the number correct; unused values add zero.
        """
        total_accuracy = 0
        for val in self.F.get(feature):
                c1p = self._cond_prob(feature, val, -1)
                c2p = self._cond_prob(feature, val, 1)
                if c1p == c2p: #equally probable
                    total_accuracy += 0.5 * self._probX(feature, val)
                if c1p > c2p: #-1 majority
                    total_accuracy += c1p * self._probX(feature, val)
                else: #1 majority
                    total_accuracy += c2p * self._probX(feature, val)
        return total_accuracy        

    def best_feature(self) -> str:
        """Return the feature with greatest information gain, including zero gains."""
        gains = {}
        for feature in self.F:
            gains[feature] = self.information_gain(feature)
        return max(gains, key=gains.get)

    def best_feature_by_accuracy(self) -> str:
        """Return the feature with greatest single-feature training accuracy."""
        accuracy = {}
        for feature in self.F:
            accuracy[feature] = self.classification_accuracy(feature)
        return max(accuracy, key=accuracy.get)

    # Provided compatibility wrappers for the original lab's helper names.
    def _entropy(self) -> float:
        return self.entropy()

    def _full_cond_entropy(self, feature: str) -> float:
        return self.conditional_entropy(feature)

    def _info_gain(self, feature: str) -> float:
        return self.information_gain(feature)

    def classification_error(self, feature: str) -> float:
        return 1.0 - self.classification_accuracy(feature)
