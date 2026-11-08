"""Chronology, calibration, decision-time availability and selector checks."""
import unittest

import numpy as np
import pandas as pd

from scripts.run_rq2_buffered_gate import (GATES, QGRID, KGRID, buffered_rankings,
    calibration_rows, check_training_chronology, estimate_margins, rank_buffered, select_parameters)


class BufferedGateTests(unittest.TestCase):
    def calibration(self):
        rows = []
        for h in (15, 30):
            for i, current in enumerate((1., 2., 3., 4.)):
                t = pd.Timestamp('2026-02-01') + pd.Timedelta(minutes=30*i)
                rows.append(dict(decision_time=t, target_time=t+pd.Timedelta(minutes=h), horizon_min=h,
                    carpark_id='P', split='validation', forecast_label_eligible=True, current_is_valid=True,
                    actual_is_observed=True, M1=.5, total_lots=10., current_lots_available=current,
                    actual_total_lots=10., actual_lots_available=0.))
        return pd.DataFrame(rows)

    def frame(self):
        t = pd.Timestamp('2026-06-15')
        return pd.DataFrame(dict(carpark_id=['near','middle','far'], distance_m=[100.,200.,300.],
            total_lots=[10.,20.,50.], current_lots_available=[2.,3.,25.], forecast_occupancy=[.7,.5,.2],
            actual_total_lots=[1000.,1.,1.], actual_lots_available=[0.,0.,0.], horizon_min=[15]*3,
            decision_time=[t]*3, target_time=[t+pd.Timedelta(minutes=15)]*3))

    def margins(self, margin=0.):
        return pd.DataFrame([dict(carpark_id=p, horizon_min=15, strategy=s, quantile=.95,
                                 margin_spaces=margin) for p in ['near','middle','far'] for s in GATES])

    def selection_frame(self):
        records = []
        for h in (15,30):
            records.append(dict(strategy='C0', horizon_min=h, quantile=np.nan, threshold=np.nan,
                matched_cases=100, top1_full_count=2, top1_low_spaces_count=3, mean_distance_m=260.))
            for q in QGRID:
                for k in KGRID:
                    records.append(dict(strategy=GATES[0], horizon_min=h, quantile=q, threshold=k,
                        matched_cases=100, top1_full_count=3 if k<5 else 1,
                        top1_low_spaces_count=4 if k<5 else 2, mean_distance_m=200.+k))
        return pd.DataFrame(records).assign(split='late_validation', cohort='open_at_departure')

    def test_higher_quantile_nonnegative_and_missing_park_rejected(self):
        samples=self.calibration(); margins=estimate_margins(samples,['P'])
        self.assertEqual(margins.loc[margins.strategy.eq(GATES[0]),'margin_spaces'].unique().tolist(),[4.])
        self.assertEqual(margins.loc[margins.strategy.eq(GATES[1]),'margin_spaces'].unique().tolist(),[5.])
        samples['actual_lots_available']=10.
        self.assertTrue(estimate_margins(samples,['P']).margin_spaces.eq(0).all())
        with self.assertRaises(ValueError): estimate_margins(samples,['P','MISSING'])

    def test_calibration_embargo_same_segment_and_test_excluded(self):
        data=self.calibration()
        extra=data.iloc[[0]].copy();extra['decision_time']=pd.Timestamp('2026-03-31 23:30')
        extra['target_time']=pd.Timestamp('2026-03-31 23:45')
        ineligible=data.iloc[[0]].assign(forecast_label_eligible=False)
        test=data.iloc[[0]].assign(split='test')
        kept=calibration_rows(pd.concat([data,extra,ineligible,test],ignore_index=True))
        self.assertEqual(len(kept),len(data))
        self.assertTrue(kept.decision_time.lt(pd.Timestamp('2026-03-31 23:30')).all())

    def test_training_target_strictly_precedes_calibration(self):
        audit={'horizons':{str(h):{'train':{'last_decision_time':'2026-01-16T23:15:00'}} for h in (15,30)}}
        check_training_chronology(audit)
        audit['horizons']['30']['train']['last_decision_time']='2026-01-16T23:30:00'
        with self.assertRaises(ValueError): check_training_chronology(audit)

    def test_forecast_uses_current_capacity_and_gate_boundary_is_inclusive(self):
        rows=self.frame(); margins=self.margins()
        ranked=rank_buffered(rows,GATES[1],.95,3,margins)
        self.assertEqual(ranked.carpark_id.tolist(),['near','middle','far'])
        self.assertAlmostEqual(ranked.estimated_free.iloc[0],3.)
        rows['actual_total_lots']=5000.;rows['actual_lots_available']=4000.
        again=rank_buffered(rows,GATES[1],.95,3,margins)
        pd.testing.assert_series_equal(ranked.estimated_free,again.estimated_free)

    def test_actual_departure_eligibility_preserved_despite_zero_buffer(self):
        rows=self.frame();margins=self.margins(100.)
        ranks=buffered_rankings(rows,.95,3,'open_at_departure',margins)
        self.assertEqual(len(ranks),6)
        self.assertTrue(ranks.candidate_count.eq(3).all())
        self.assertTrue(ranks.estimated_free.eq(0).all())
        self.assertTrue(ranks.current_lots_available.gt(0).all())
        self.assertFalse(ranks.gate_qualifies.any())
        self.assertEqual(ranks.loc[ranks['rank'].eq(1),'carpark_id'].tolist(),['near','near'])
        rows.loc[0,'current_lots_available']=0.
        ranks=buffered_rankings(rows,.95,3,'open_at_departure',margins)
        self.assertTrue(ranks.candidate_count.eq(2).all())

    def test_no_qualifier_falls_back_to_largest_buffered_estimate(self):
        rows=self.frame();rows['current_lots_available']=[1.,2.,2.]
        self.assertEqual(rank_buffered(rows,GATES[0],.95,3,self.margins()).carpark_id.tolist(),['middle','far','near'])

    def test_shared_selection_both_horizons_and_fixed_fallback(self):
        rows=self.selection_frame()
        q,k,table=select_parameters(rows)
        self.assertEqual((q,k),(.90,5))
        rows.loc[rows.horizon_min.eq(30)&rows.threshold.eq(5),'top1_full_count']=3
        self.assertEqual(select_parameters(rows)[:2],(.90,10))
        rows.loc[rows.strategy.eq(GATES[0]),'top1_full_count']=4
        q,k,table=select_parameters(rows)
        self.assertEqual((q,k),(.95,3));self.assertTrue(table.diagnostic_fallback_used.all())

    def test_selector_rejects_forecast_or_test_rows(self):
        rows=self.selection_frame();rows.loc[0,'split']='test'
        with self.assertRaises(ValueError):select_parameters(rows)
        rows=self.selection_frame();rows.loc[1,'strategy']=GATES[1]
        with self.assertRaises(ValueError):select_parameters(rows)


if __name__=='__main__':unittest.main()
