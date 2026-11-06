import unittest
import pandas as pd

from forecast import split_times
from scripts.run_holiday_holdout import (add_target_calendar, holiday_calendar, validate_audit,
                                        SOURCE_START, SOURCE_END, VALIDATION_START, TEST_START)
from pathlib import Path


class HolidayHoldoutTests(unittest.TestCase):
    def test_midnight_arrival_belongs_to_target_holiday_date(self):
        calendar=pd.DataFrame({"date":pd.to_datetime(["2026-01-01"]),"holiday":["New Year"],
                               "episode_id":["2026-01-01"]})
        frame=pd.DataFrame({"decision_time":pd.to_datetime(["2025-12-31 23:45","2026-01-01 23:45"]),
                            "target_time":pd.to_datetime(["2026-01-01 00:00","2026-01-02 00:00"]),
                            "is_holiday":[1,0]})
        result=add_target_calendar(frame,calendar)
        self.assertEqual(result.target_is_holiday.tolist(),[True,False])
        self.assertEqual(result.target_date.iloc[0],pd.Timestamp("2026-01-01"))

    def test_source_calendar_forms_twelve_test_dates_in_nine_episodes(self):
        calendar=holiday_calendar(Path(__file__).resolve().parents[1]/"data/sg_public_holidays_2020_2027.csv")
        test=calendar.loc[calendar.date.ge(TEST_START)&calendar.date.le(SOURCE_END)]
        self.assertEqual(len(test),12)
        self.assertEqual(test.episode_id.nunique(),9)
        for _,episode in test.groupby("episode_id"):
            self.assertTrue(episode.date.diff().dropna().eq(pd.Timedelta(days=1)).all())

    def test_fixed_boundaries_embargo_both_label_horizons(self):
        times=pd.Series(pd.date_range(SOURCE_START,SOURCE_END,freq="15min"))
        split=pd.Series(split_times(times,(VALIDATION_START,TEST_START)),index=times.index)
        for period,boundary in (("train",VALIDATION_START),("validation",TEST_START)):
            latest=times.loc[split.eq(period)].max()
            self.assertLess(latest+pd.Timedelta(minutes=30),boundary)
            self.assertTrue(split.loc[times.ge(boundary-pd.Timedelta(minutes=30))&times.lt(boundary)].eq("embargo").all())

    def test_reuse_audit_rejects_wrong_boundaries(self):
        audit={"first_time":SOURCE_START.isoformat(),"last_time":SOURCE_END.isoformat(),"horizons":{}}
        for h in ("15","30"):
            audit["horizons"][h]={}
            for split,start,last in (("train",SOURCE_START,VALIDATION_START-pd.Timedelta(minutes=45)),
                                     ("validation",VALIDATION_START,TEST_START-pd.Timedelta(minutes=45)),
                                     ("test",TEST_START,SOURCE_END)):
                audit["horizons"][h][split]={"first_decision_time":start.isoformat(),"last_decision_time":last.isoformat(),
                                             "holiday_days":1,"unknown_holiday_labels":0}
        validate_audit(audit)
        audit["horizons"]["30"]["test"]["first_decision_time"]="2026-06-12T00:00:00"
        with self.assertRaises(ValueError): validate_audit(audit)


if __name__=="__main__":
    unittest.main()
