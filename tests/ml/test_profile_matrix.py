"""Inert model fixtures verify matrix quality denominators and rotating paired execution."""
from contextlib import redirect_stdout
from io import StringIO
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from tests.ml.test_candidate_benchmark import frozen_fixture,fake_predictors
from infra.portable.profile_matrix import measure

class MatrixEvidence(unittest.TestCase):
    def test_complete_denominators_and_model_failure_are_preserved(self):
        payload,annotation=frozen_fixture();base,student=fake_predictors(annotation,wrong_image=83)
        profiles=[{'name':'baseline','directory':'base'},{'name':'compact','directory':'student'}]
        with patch('ml.inference.onnx_runtime.OnnxCascadePredictor',base),patch('ml.inference.candidate_onnx_runtime.CandidateOnnxPredictor',student),patch.dict(sys.modules,{'onnxruntime':SimpleNamespace(__version__='fixture')}),redirect_stdout(StringIO()):
            report=measure(profiles,payload,annotation)
        self.assertFalse(report['partial']);self.assertFalse(report['testOpened']);self.assertFalse(report['trainingStarted'])
        self.assertEqual(report['summaries']['compact']['all150']['successes'],149)
        self.assertEqual(report['summaries']['compact']['all150']['wrongAccepts'],1)
        self.assertEqual(report['summaries']['compact']['thirteenFourteen45']['timedImages'],45)
        self.assertEqual(report['strictHistoricalGate']['compact']['newWrongAcceptIds'],[83])
    def test_baseline_is_required(self):
        with self.assertRaises(ValueError):measure([{'name':'other','directory':'x'}],[],{})
if __name__=='__main__':unittest.main()
