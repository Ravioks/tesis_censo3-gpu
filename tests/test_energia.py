import http.server
import json
import threading
import time

import pytest

import medidores_energia as me

LHM = {"id": 0, "Text": "Sensor", "Value": "", "Children": [{"id": 1, "Text": "DESKTOP", "Value": "", "Children": [
    {"id": 2, "Text": "Intel Core i5-9400", "Value": "", "Children": [
        {"id": 3, "Text": "Powers", "Value": "", "Children": [
            {"id": 4, "Text": "CPU Package", "Value": "35,2 W", "SensorId": "/intelcpu/0/power/0", "Type": "Power",
             "Children": []},
            {"id": 5, "Text": "CPU Cores", "Value": "30,1 W", "SensorId": "/intelcpu/0/power/1", "Type": "Power",
             "Children": []}]},
        {"id": 6, "Text": "Temperatures", "Value": "", "Children": [
            {"id": 7, "Text": "CPU Package", "Value": "55,0 °C", "SensorId": "/intelcpu/0/temperature/6",
             "Type": "Temperature", "Children": []}]}]},
    {"id": 8, "Text": "NVIDIA GeForce RTX 5060", "Value": "", "Children": [
        {"id": 9, "Text": "Powers", "Value": "", "Children": [
            {"id": 10, "Text": "GPU Package", "Value": "18.5 W", "SensorId": "/gpu-nvidia/0/power/0", "Type": "Power",
             "Children": []}]}]}]}]}


def sin_ids(nodo):
    n = {k: v for k, v in nodo.items() if k not in ("SensorId", "Type")}
    n["Children"] = [sin_ids(h) for h in nodo.get("Children", [])]
    return n


def test_integrar_constante_y_rampa():
    muestras = [(t / 10, 50.0) for t in range(0, 101)]
    r = me.integrar(muestras, 2.0, 7.0)
    assert r["julios"] == pytest.approx(250.0)
    assert r["watts"] == pytest.approx(50.0)
    rampa = [(t / 10, t / 10) for t in range(0, 101)]
    assert me.integrar(rampa, 0.0, 10.0)["julios"] == pytest.approx(50.0)
    assert me.integrar(rampa, 2.05, 2.95)["julios"] == pytest.approx((2.95 ** 2 - 2.05 ** 2) / 2)


def test_contador_con_desborde():
    valores = iter([90.0, 5.0])
    c = me.Contador("x", lambda: next(valores), rango=100.0)
    c.marcar("a")
    time.sleep(0.01)
    c.marcar("b")
    assert c.energia_entre("a", "b")["julios"] == pytest.approx(15.0)


@pytest.mark.parametrize("arbol", [LHM, sin_ids(LHM)], ids=["con_ids", "sin_ids"])
def test_sensores_lhm_eleccion_y_coma_decimal(arbol):
    sensores = me.sensores_lhm(arbol)
    cpu = me.elegir_sensor(sensores, "cpu")
    gpu = me.elegir_sensor(sensores, "gpu")
    assert cpu["texto"] == "CPU Package" and me.valor_watts(cpu) == pytest.approx(35.2)
    assert gpu["texto"] == "GPU Package" and me.valor_watts(gpu) == pytest.approx(18.5)
    assert me.elegir_sensor(sensores, "cpu", "CPU Cores")["texto"] == "CPU Cores"


def test_medidores_con_servidor_lhm_local():
    cuerpo = json.dumps(LHM).encode()

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(cuerpo)

        def log_message(self, *a):
            pass
    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/data.json"
    m = me.Medidores("lhm", "lhm", url_lhm=url, periodo_lhm=0.02)
    m.iniciar()
    ta = m.marcar("a")
    time.sleep(0.5)
    tb = m.marcar("b")
    m.parar()
    srv.shutdown()
    assert m.energia("cpu", "a", "b", ta, tb)["watts"] == pytest.approx(35.2)
    assert m.energia("gpu", "a", "b", ta, tb)["watts"] == pytest.approx(18.5)
    assert m.info()["descripcion"]["cpu"]["fuente"] == "lhm"
