from typing import NamedTuple

import math
from ..generator import symbol, op, Invocation, Symbol

__all__ = [
  'scale_grammar',
  'ConstantGrammar'
]

def scale_grammar(
  small: Symbol, normal: Symbol, large: Symbol,
  small_positive: Symbol, normal_positive: Symbol, large_positive: Symbol,
  small_constant: Invocation | Symbol, normal_constant: Invocation | Symbol, large_constant: Invocation | Symbol,
  small_positive_constant: Invocation | Symbol,
  normal_positive_constant: Invocation | Symbol,
  large_positive_constant: Invocation | Symbol,
):

  return {
    normal.when('complexity > 0'): {
      normal('complexity - 1'): '2 * complexity',

      ### const + normal
      small('complexity - 2') + normal('complexity - 1') + op('add'): 1.0,
      small_constant + normal('complexity - 1') + op('add'): 1.0,

      ### normal - normal
      normal('complexity - 2') + normal('complexity - 1') + op('sub'): 0.5,
      normal('complexity - 1') + normal('complexity - 2') + op('sub'): 0.5,

      ### normal - const
      small_constant + normal('complexity - 1') + op('sub'): 0.5,
      normal('complexity - 1') + small_constant + op('sub'): 0.5,

      ### normal - small
      small('complexity - 2') + normal('complexity - 1') + op('sub'): 0.5,
      small('complexity - 1') + normal('complexity - 2') + op('sub'): 0.5,

      normal('complexity - 1') + small('complexity - 2') + op('sub'): 0.5,
      normal('complexity - 2') + small('complexity - 1') + op('sub'): 0.5,

      ### const * expr
      small_constant + large('complexity - 1') + op('mul'): 0.5,
      large_constant + small('complexity - 1') + op('mul'): 0.5,
      normal_constant + normal('complexity - 1') + op('mul'): 0.5,

      ### expr * expr
      small('complexity - 2') + large('complexity - 1') + op('mul'): 0.5,
      small('complexity - 1') + large('complexity - 2') + op('mul'): 0.5,

      ### expr / expr
      small_positive_constant + normal_positive('complexity - 1') + op('add') + normal('complexity - 2') + op('div'): 0.25,
      normal_positive_constant + normal_positive('complexity - 1') + op('add') + normal('complexity - 2') + op('div'): 0.25,
      small_positive_constant + normal_positive('complexity - 2') + op('add') + normal('complexity - 1') + op('div'): 0.25,
      normal_positive_constant + normal_positive('complexity - 2') + op('add') + normal('complexity - 1') + op('div'): 0.25,
      normal_positive_constant + large_positive('complexity - 1') + op('add') + large('complexity - 2') + op('div'): 0.25,
      normal_positive_constant + large_positive('complexity - 2') + op('add') + large('complexity - 1') + op('div'): 0.25,

      ### exp
      small('complexity - 1') + op('exp'): 1.0,

      ### log
      small_positive('complexity - 2') + large_positive('complexity - 1') + op('add') + op('log'): 1.0,
      small_positive_constant + large_positive('complexity - 1') + op('add') + op('log'): 1.0,

      ### sqrt
      normal_positive('complexity - 1') + op('sqrt'): 1.0,

      ### softplus
      normal('complexity - 1') + op('softplus'): 1.0,
    },
    normal_positive.when('complexity > 0'): {
      small_positive_constant + normal_positive('complexity - 1') + op('add'): 1.0,
      small_positive('complexity - 1') + normal_positive('complexity - 1') + op('add'): 1.0,

      small_positive('complexity - 2') + large_positive('complexity - 1') + op('mul'): 0.5,
      small_positive('complexity - 1') + large_positive('complexity - 2') + op('mul'): 0.5,

      small_positive_constant + large_positive('complexity - 1') + op('mul'): 0.5,
      large_positive_constant + small_positive('complexity - 1') + op('mul'): 0.5,

      small_positive_constant + normal_positive('complexity - 1') + op('add') + normal_positive('complexity - 2') + op('div'): 0.25,
      normal_positive_constant + normal_positive('complexity - 1') + op('add') + normal_positive('complexity - 2') + op('div'): 0.25,
      small_positive_constant + normal_positive('complexity - 2') + op('add') + normal_positive('complexity - 1') + op('div'): 0.25,
      normal_positive_constant + normal_positive('complexity - 2') + op('add') + normal_positive('complexity - 1') + op('div'): 0.25,
      normal_positive_constant + large_positive('complexity - 1') + op('add') + large_positive('complexity - 2') + op('div'): 0.25,
      normal_positive_constant + large_positive('complexity - 2') + op('add') + large_positive('complexity - 1') + op('div'): 0.25,

      small('complexity - 1') + op('exp'): 1.0,
      op('const')(1.0) + large_positive('complexity - 1') + op('add') + op('log'): 1.0,
      normal_positive('complexity - 1') + op('sqrt'): 1.0,
      large_positive('complexity - 1') + op('sqrt'): 1.0,

      normal('complexity - 1') + op('softplus'): 1.0,
    },
    small.when('complexity > 0'): {
      small('complexity - 1'): '2 * complexity',

      small('complexity - 2') + small('complexity - 1') + op('sub'): 0.5,
      small('complexity - 1') + small('complexity - 2') + op('sub'): 0.5,

      small_constant + normal('complexity - 1') + op('mul'): 0.5,

      small_positive_constant + large_positive('complexity - 2') + op('add') + normal('complexity - 1') + op('div'): 0.33,
      small_positive_constant + large_positive('complexity - 2') + op('add') + normal('complexity - 2') + op('div'): 0.33,
      normal_positive_constant + large_positive('complexity - 2') + op('add') + normal('complexity - 2') + op('div'): 0.33,
      normal_positive_constant + large_positive('complexity - 2') + op('add') + normal('complexity - 2') + op('div'): 0.33,
      large_positive_constant + large_positive('complexity - 2') + op('add') + normal('complexity - 2') + op('div'): 0.33,
      large_positive_constant + large_positive('complexity - 2') + op('add') + normal('complexity - 2') + op('div'): 0.33,

      small_positive_constant + normal_positive('complexity - 2') + op('add') + small('complexity - 1') + op('div'): 0.33,
      small_positive_constant + normal_positive('complexity - 1') + op('add') + small('complexity - 2') + op('div'): 0.33,
      normal_positive_constant + normal_positive('complexity - 2') + op('add') + small('complexity - 1') + op('div'): 0.33,
      normal_positive_constant + normal_positive('complexity - 1') + op('add') + small('complexity - 2') + op('div'): 0.33,

      small_positive('complexity - 1') + op('sqrt'): 1.0,

      large_positive('complexity - 1') + op('neg') + op('exp'): 1.0,
      normal_positive('complexity - 1') + op('neg') + op('exp'): 1.0,

      # normal_positive_constant + normal_positive('complexity - 1') + op('add') + op('log'): 1.0,
      # large_positive_constant + normal_positive('complexity - 1') + op('add') + op('log'): 1.0,

      small('complexity - 1') + op('square'): 1.0,
      small('complexity - 1') + op('cube'): 1.0,

      small('complexity - 1') + op('softplus'): 1.0,
      small('complexity - 1') + op('tanh'): 1.0,
      small('complexity - 1') + op('erf'): 1.0,

      small('complexity - 1') + op('sigmoid'): 1.0,
      normal('complexity - 1') + op('sigmoid'): 1.0,
      large('complexity - 1') + op('sigmoid'): 1.0,
    },
    small_positive.when('complexity > 0') : {
      small_positive_constant + small_positive('complexity - 2') + op('add'): 3.0,

      small_positive_constant + normal_positive('complexity - 1') + op('add') + small_positive('complexity - 2') + op('div'): 0.25,
      small_positive_constant + normal_positive('complexity - 2') + op('add') + small_positive('complexity - 1') + op('div'): 0.25,
      normal_positive_constant + normal_positive('complexity - 1') + op('add') + small_positive('complexity - 2') + op('div'): 0.25,
      normal_positive_constant + normal_positive('complexity - 2') + op('add') + small_positive('complexity - 1') + op('div'): 0.25,

      small_positive('complexity - 1') + op('sqrt'): 1.0,

      large_positive('complexity - 1') + op('neg') + op('exp'): 1.0,
      op('const')(1.0) + normal_positive('complexity - 1') + op('add') + op('log'): 1.0,
      small('complexity - 1') + op('square'): 1.0,
      small_positive('complexity - 1') + op('cube'): 1.0,

      small('complexity - 1') + op('softplus'): 1.0,

      small('complexity - 1') + op('sigmoid'): 1.0,
      normal('complexity - 1') + op('sigmoid'): 1.0,
      large('complexity - 1') + op('sigmoid'): 1.0,
    },
    large.when('complexity > 0'): {
      large('complexity - 1'): 2.0,

      small_constant + large('complexity - 1') + op('add'): 1.0,
      normal_constant + large('complexity - 1') + op('add'): 1.0,

      normal('complexity - 2') + large('complexity - 1') + op('add'): 1.0,
      normal('complexity - 1') + large('complexity - 2') + op('add'): 1.0,

      small('complexity - 2') + large('complexity - 1') + op('add'): 1.0,
      small('complexity - 1') + large('complexity - 2') + op('add'): 1.0,

      normal_constant + large('complexity - 1') + op('add'): 1.0,

      large('complexity - 2') + large('complexity - 1') + op('sub'): 0.5,
      large('complexity - 1') + large('complexity - 2') + op('sub'): 0.5,

      normal('complexity - 2') + normal('complexity - 1') + op('mul'): 0.5,

      small_positive_constant + small_positive('complexity - 1') + op('add') + normal('complexity - 2') + op('div'): 0.25,
      small_positive_constant + small_positive('complexity - 2') + op('add') + normal('complexity - 1') + op('div'): 0.25,
      small_positive_constant + normal_positive('complexity - 1') + op('add') + large('complexity - 2') + op('div'): 0.25,
      small_positive_constant + normal_positive('complexity - 2') + op('add') + large('complexity - 1') + op('div'): 0.25,

      large_positive('complexity - 1') + op('sqrt'): 1.0,

      small_positive_constant + small_positive('complexity - 1') + op('add') + op('log'): 1.0,
      small_positive_constant + small_positive('complexity - 1') + op('add') + op('log'): 1.0,

      normal('complexity - 1') + op('exp'): 1.0,
      normal('complexity - 1') + op('square'): 1.0,

      large('complexity - 1') + op('softplus'): 1.0,
    },
    large_positive.when('complexity > 0'): {
      small_positive_constant + large_positive('complexity - 1') + op('add'): 1.0,
      normal_positive_constant + large_positive('complexity - 1') + op('add'): 1.0,

      normal_positive('complexity - 2') + large_positive('complexity - 1') + op('add'): 1.0,
      small_positive('complexity - 2') + large_positive('complexity - 1') + op('add'): 1.0,

      normal_positive('complexity - 2') + normal_positive('complexity - 1') + op('mul'): 0.5,

      small_positive_constant + small_positive('complexity - 1') + op('add') + normal_positive('complexity - 2') + op('div'): 0.25,
      small_positive_constant + small_positive('complexity - 2') + op('add') + normal_positive('complexity - 1') + op('div'): 0.25,
      small_positive_constant + normal_positive('complexity - 1') + op('add') + large_positive('complexity - 2') + op('div'): 0.25,
      small_positive_constant + normal_positive('complexity - 2') + op('add') + large_positive('complexity - 1') + op('div'): 0.25,

      large_positive('complexity - 1') + op('sqrt'): 1.0,

      normal('complexity - 1') + op('exp'): 1.0,
      normal('complexity - 1') + op('square'): 1.0,

      large('complexity - 1') + op('softplus'): 1.0,
    },
  }

class ConstantGrammar(NamedTuple):
  normal: Symbol = symbol('normal_constant')('complexity')
  small: Symbol = symbol('small_constant')('complexity')
  large: Symbol = symbol('large_constant')('complexity')

  normal_positive: Symbol = symbol('normal_positive_constant')('complexity')
  small_positive: Symbol = symbol('small_positive_constant')('complexity')
  large_positive: Symbol = symbol('large_positive_constant')('complexity')

  normal_positive_value: Symbol = symbol('normal_positive_value')()
  small_positive_value: Symbol = symbol('small_positive_value')()
  large_positive_value: Symbol = symbol('large_positive_value')()

  def rules(self):
    return {
      **scale_grammar(
        self.small, self.normal, self.large, self.small_positive, self.normal_positive, self.large_positive,
        self.small_positive_value, self.normal_positive_value, self.large_positive_value,
        self.small_positive_value, self.normal_positive_value, self.large_positive_value,
      ),
      self.small: {
        self.small_positive_value: 1.0,
        self.small_positive_value + op('neg'): 1.0
      },
      self.normal: {
        self.normal_positive_value: 1.0,
        self.normal_positive_value + op('neg'): 1.0,
  
      },
      self.large: {
        self.large_positive_value: 1.0,
        self.large_positive_value + op('neg'): 1.0
      },
      self.small_positive: self.small_positive_value,
      self.normal_positive: self.normal_positive_value,
      self.large_positive: self.large_positive_value,
  
      self.small_positive_value: {
        op('const')(0.5): 1.0,
        op('const')(math.sqrt(0.5)): 1.0,
        op('const')(math.exp(-1)): 1.0,
        op('const')(math.log(2)): 1.0,

        self.small_positive_value + self.large_positive_value + op('add') + op('const')(1.0) + op('div'): 1.0,
        self.normal_positive_value + self.large_positive_value + op('add') + op('const')(1.0) + op('div'): 1.0,
      },
      self.normal_positive_value: {
        op('const')(1.0): 1.0,
        op('const')(math.pi): 1.0,
        op('const')(math.e): 1.0,
        op('const')(2.0): 1.0,
        op('const')(3.0): 1.0,
      },
      self.large_positive_value: {
        self.normal_positive_value + self.normal_positive_value + op('mul'): 1.0,
        self.normal_positive_value + op('square'): 1.0,
      }
    }

class VariableGrammar(NamedTuple):
  normal = symbol('normal')('complexity')
  small = symbol('small')('complexity')
  large = symbol('large')('complexity')

  normal_positive = symbol('normal_positive')('complexity')
  small_positive = symbol('small_positive')('complexity')
  large_positive = symbol('large_positive')('complexity')

  def rules(
    self, small_value, normal_value, large_value,
    small_positive_value, normal_positive_value, large_positive_value
  ):
    return {
      **scale_grammar(
        self.small, self.normal, self.large, self.small_positive, self.normal_positive, self.large_positive,
        small_positive_value, normal_positive_value, large_positive_value,
        small_positive_value, normal_positive_value, large_positive_value,
      ),
      self.small: small_value,
      self.normal: normal_value,
      self.large: large_value,
      self.small_positive: small_positive_value,
      self.normal_positive: normal_positive_value,
      self.large_positive: large_positive_value,
    }