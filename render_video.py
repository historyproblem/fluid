import numpy as np
import imageio.v2 as imageio

from vispy import app, scene
from tmp_test import PBFSimulation

app.use_app('pyqt6')

class Recorder(scene.SceneCanvas):
    def __init__(self, sim: PBFSimulation, fps: int = 60, duration: float = 100.0, outfile: str = "sph_record.mp4"):
        super().__init__(size=(1000, 650), bgcolor="white", show=False)
        self.unfreeze()
        self.sim = sim
        self.duration = duration
        self.target_fps = fps
        self.frames = int(fps * duration)
        self.outfile = outfile

        self.view = self.central_widget.add_view()
        self.view.camera = scene.cameras.PanZoomCamera(aspect=1)
        if hasattr(self.sim, "bounds"):
            xmin, xmax, ymin, ymax = self.sim.bounds
        else:
            xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0
        self.view.camera.set_range(x=(xmin, xmax), y=(ymin, ymax))

        pos = np.asarray(self.sim.positions, dtype=float)
        if pos.ndim != 2 or pos.shape[1] != 2:
            raise ValueError(f"PBFSimulation.positions должен быть формы (N,2); сейчас: {pos.shape}")
        self.markers = scene.visuals.Markers(parent=self.view.scene)
        self.markers.set_data(pos, face_color="navy", size=4.0, edge_width=0)
        self.freeze()

    def record(self):
        writer = imageio.get_writer(self.outfile, fps=self.fps, codec="libx264", quality=8)
        for f in range(self.frames):
            self.sim.step()
            self.markers.set_data(self.sim.positions, face_color="navy", size=4.0, edge_width=0)

            img = self.render()
            writer.append_data(img)
            if f % self.target_fps == 0:
                print(f"[INFO] Записано {f/self.target_fps:.1f} сек...")
        writer.close()
        print(f"[INFO] Видео сохранено в {self.outfile}")

def main():
    sim = PBFSimulation(
        # bounds=(0, 100, 0, 100),
        bounds=(0, 15, 0, 10),
        N=256,
        gravity=(0.0, -4387.256940771852),
        # gravity=(0.0, -500.0),
        rest_density= 4000.0,
        # rest_density=1000.0,
        # mass_coeff=2.55,
        mass_coeff=2.5694176847141144,
        h_dx_ratio=2.5,
        # h_dx_ratio=1.4514124356223201,
        iterations=20,
        # nu_xsph=1.0,
        nu_xsph= 0.7800384056051926,
        scorr_k=0.001,
        scorr_n=5,
        steps=600,
        seed=42)
    rec = Recorder(sim, fps=60, duration=10.0, outfile="sph_record.mp4")
    rec.record()

if __name__ == "__main__":
    main()
