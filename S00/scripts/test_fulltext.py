import unittest
from fulltext_b0 import merge_segments

class FulltextTests(unittest.TestCase):
    def test_all_segments_in_time_order_without_mutating_source(self):
        segments=[{'start':2.,'end':3.,'text':'后半句。'},{'start':0.,'end':1.,'text':'前半句，'}]
        text,ordered=merge_segments(segments)
        self.assertEqual(text,'前半句，后半句。')
        self.assertEqual(ordered[0]['start'],0.)
        self.assertEqual(segments[0]['start'],2.)
    def test_single_segment_unchanged(self):
        self.assertEqual(merge_segments([{'start':0.,'end':1.,'text':' 原文。'}])[0],' 原文。')
    def test_invalid_or_empty_rejected(self):
        for segments in ([],[{'start':0.,'end':1.,'text':' '}],[{'start':0.,'end':float('nan'),'text':'text'}],[{'start':2.,'end':1.,'text':'text'}],[{'start':0.,'end':1.,'text':'[ERROR]'}]):
            with self.subTest(segments=segments),self.assertRaises(ValueError):merge_segments(segments)

if __name__=='__main__':unittest.main(verbosity=2)
