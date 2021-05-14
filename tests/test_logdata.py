from src.log_data import LogData
import pytest

@pytest.mark.skip()
def test_logdata(tmp_path):
    logdata = LogData(folder=tmp_path)
    print (tmp_path)
    logdata.add_logfile("nn_loss", ["nn_loss.csv", ['iter', 'loss', 'value_loss', 'prob_loss']])
    assert 0

