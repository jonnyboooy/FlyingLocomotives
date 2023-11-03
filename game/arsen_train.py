import math
import random
from game.locator import Locator
from game.geometry import Figures


class Train:
    def __init__(self, x0: float, y0: float, alpha0: float, v_max: float, locator: Locator):
        self.alpha = alpha0
        self.x = x0
        self.y = y0

        self.v_max = v_max
        self.locator = locator

        self.v = 5
        self.shape = None
        self.distance = None
        self.maps = []
        self.auto = True

        self.rotation = True
        self.points_buffer = []
        self.points_counter = 0
        self.alpha_buffer = 0
        self.move_counter = 0

        self.figures = Figures()

        # цвета:
        self.red = (255, 0, 0)
        self.green = [(0, 255, 0)]
        self.blue = [(0, 0, 255)]

    def update(self, x: float, y: float):
        if self.auto:
            self.x = x
            self.y = y

        updated_data = self.locator.measurement

        if updated_data['query']:
            _x, _y, _alpha = updated_data['query'][0]
            self.distance = updated_data['distance']

            if self.distance:
                new_touch_point = (
                    _x + self.distance * math.cos(_alpha),
                    _y + self.distance * math.sin(_alpha)
                )

                if not self.figures.is_point_on_the_border(new_touch_point):
                    self.figures.points.append(new_touch_point)
                    self.points_buffer.append(new_touch_point)

                if len(self.points_buffer) == 4:
                    self.figures.get_figures(self.points_buffer, (self.x, self.y))
                    self.points_buffer.pop(0)
            else:
                if self.points_buffer:
                    self.points_buffer = []
        else:
            self.distance = None

    def info(self):
        # TODO!
        # red = [(255, 0, 0)]
        # line1 = [(100, 200), (400, 300)]
        # line2 = [(150, 250), (150, 350), (250, 350)]
        # line3 = [(0, 0), (500, 500), (0, 1000)]
        # circle1 = ((100, 200), 20)  # (point, radius)
        # circle2 = ((200, 400), 30)  # (point, radius)
        # circle3 = ((400, 600), 40)  # (point, radius)

        # [[(89.64345646009977, 697.6717575270931), (1123.8407389109825, 697.6717575270931)], [
        #     (22.412144926119765, 30.071195292699258), (22.412144926119765, 649.9288047073061)], [
        #     (36.5991221988523, 22.820110944117516), (1214.0673274858311, 22.820110944117516)], [
        #     (479.8490537360184, 142.62892959833295), (563.8538098058478, 278.34977067333523)], [
        #     (1237.8218308940243, 21.74339504852115), (1237.8218308940243, 613.2831517407436)], [
        #     (410.14817508698434, 411.8088487572956), (576.7571922345039, 308.6340146049854)], [
        #     (304.7751471577461, 240.27666384640162), (369.82246073841986, 200.38688189044757)], [
        #     (253.29189959234492, 348.20239941716136), (322.99587661851115, 460.9664026418425)], [
        #     (1214.0673274858311, 22.820301903430916), (1171.1781816969353, 22.819839036519888)]]

        figures = {
            "lines": self.figures.lines,
            "circles": self.figures.circles,
            "points": self.figures.points
        }

        return {
            'params': (self.x, self.y, self.v, self.alpha),
            'maps': figures
        }

    def processing(self):
        if self.auto:
            self.autopilot()

        self.locator.make_query(self.x, self.y, self.alpha)

    def manual_update(self, x: float, y: float, alpha: float):
        if not self.auto:
            self.x += x
            self.y += y
            self.alpha += alpha

        self.locator.make_query(self.x, self.y, self.alpha)

    def autopilot(self):
        if self.move_counter % 300 == 0:
            self.rotation = True

        if self.distance:
            if not self.rotation and self.distance < 30:
                if random.randint(0, 1) == 0:
                    self.alpha += math.pi + random.uniform(- math.pi / 3, math.pi / 3)
                else:
                    self.alpha += - math.pi + random.uniform(- math.pi / 3, math.pi / 3)

        if self.rotation:
            if self.alpha_buffer > 2 * math.pi:
                if random.randint(0, 1) == 0:
                    self.alpha += 2 * math.pi / 3 + random.uniform(- math.pi / 3, math.pi / 3)
                else:
                    self.alpha += - 2 * math.pi / 3 + random.uniform(- math.pi / 3, math.pi / 3)

                self.alpha_buffer = 0
                self.rotation = False

            self.alpha += math.radians(5)
            self.alpha_buffer += math.radians(5)
        else:
            self.x += self.v * math.cos(self.alpha)
            self.y += self.v * math.sin(self.alpha)

        self.move_counter += 1
