import numpy as np
import pytest
from ..evaluator import eval_node
import json

INF285_MODEL = json.load(open('../models/INF285.json'))

def test_CC():
	model = INF285_MODEL
	ctx = model['context']
	ctx['values'] = [51.0, 31.0, 54.0, 95.0, 40.0, 73.0, 100.0, 100.0, 90.0]
	out = eval_node(model['AST'], ctx)
	assert abs(out - 70.119501) < 1e-6

def test_CC_kbest():
	model = INF285_MODEL
	ctx = model['context']
	ctx['values'] = [51.0, 31.0, 54.0, 95.0, 40.0, 73.0, 100.0, 100.0, 90.0]
	out = eval_node({
		"op": "mean",
		"args": [
			{
			"op": "k_best",
			"k": 3,
			"args": [
				{
					"op": "ref_template",
					"template": "laboratorio"
				}
			]
			}
		]
	  }, ctx)
	assert abs(out - 98.333333) < 1e-6

def test_CC_gamma():
	model = INF285_MODEL
	ctx = model['context']
	ctx['values'] = [51.0, 31.0, 54.0, 95.0, 40.0, 73.0, 100.0, 100.0, 90.0]
	out = eval_node({
        "op": "sum",
        "args": [
          {
            "op": "const",
            "value": 1.0
          },
          {
            "op": "mul",
            "args": [
              {
                "op": "ref",
                "id": 8
              },
              {
                "op": "const",
                "value": 0.001
              }
            ]
          }
        ]
      }, ctx)
	assert abs(out - 1.09) < 1e-6

def test_CC_PC():
	model = INF285_MODEL
	ctx = model['context']
	ctx['values'] = [51.0, 31.0, 54.0, 95.0, 40.0, 73.0, 100.0, 100.0, 90.0]
	out = eval_node({
            "op": "root",
            "exponent": 3,
            "args": [
              {
                "op": "mul",
                "args": [
                  {
                    "op": "ref",
                    "id": 2
                  },
                  {
                    "op": "power",
                    "exponent": 2,
                    "args": [
                      {
                        "op": "mean",
                        "args": [
                          {
                            "op": "k_best",
                            "k": 2,
                            "args": [
                              {
                                "op": "ref_template",
                                "template": "certamen"
                              }
                            ]
                          }
                        ]
                      }
                    ]
                  }
                ]
              }
            ]
          }, ctx)
	assert abs(out - 52.995312) < 1e-6