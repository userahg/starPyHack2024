import unittest
import common.starccm_macros as sm


class TestLoops(unittest.TestCase):

    def setUp(self) -> None:
        self.loop = sm.LoopComponent()

    def test_for_i_loop_0_to_3_by_1(self):
        target = "for (int i = 0; i < 4; i += 1) {\n" \
                 "    System.out.println(i);\n" \
                 "}\n"
        self.loop.loop_type = sm.LoopType.FOR_I
        self.loop.i_loop_i_name = "i"
        self.loop.i_loop_start = 0
        self.loop.i_loop_end = 4
        self.loop.i_loop_inc = 1
        self.loop.i_loop_comparator = "<"
        self.loop.add_line("System.out.println(i)")
        text = self.loop.build(0)
        self.assertEqual(text, target, "The for i 0 to 3 by 1 loop test is incorrect")

    def test_for_i_loop_3_to_0_by_1(self):
        target = "for (int i = 3; i >= 0; i += -1) {\n" \
                 "    System.out.println(i);\n" \
                 "}\n"
        self.loop.loop_type = sm.LoopType.FOR_I
        self.loop.i_loop_i_name = "i"
        self.loop.i_loop_start = 3
        self.loop.i_loop_end = 0
        self.loop.i_loop_inc = -1
        self.loop.i_loop_comparator = ">="
        self.loop.add_line("System.out.println(i)")
        text = self.loop.build(0)
        self.assertEqual(text, target, "The for i 0 to 3 by 1 loop test is incorrect")

    def test_for_all_loop(self):
        target = "for (int i : new int[] {-3, 2, -5}) {\n" \
                 "    System.out.println(i);\n" \
                 "}\n"
        self.loop.loop_type = sm.LoopType.FOR_ALL
        self.loop.for_all_loop_iterate_class = "int"
        self.loop.for_all_loop_iterate_name = "i"
        self.loop.for_all_loop_iterable = "new int[] {-3, 2, -5}"
        self.loop.add_line("System.out.println(i)")
        text = self.loop.build(0)
        self.assertEqual(text, target, "The for all loop test is incorrect")

    def test_while_loop(self):
        target = "while (i < 5) {\n" \
                 "    i++;\n" \
                 "}\n"
        self.loop.loop_type = sm.LoopType.WHILE
        self.loop.while_loop_cond = "i < 5"
        self.loop.add_line("i++")
        text = self.loop.build(0)
        self.assertEqual(text, target, "While loop test is failing")


if __name__ == "__main__":
    unittest.main()
