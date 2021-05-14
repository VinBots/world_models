import csv

class LogData:
    def __init__(self, folder="../monitoring"):

        self.folder = folder
        self.all_charts = {}

    def add_logfile (self, chart_name, data):
        """
        Adds a chart specifying a name, the name of the csv file where data will be saved
        and a list of fields (headers). The names of fields will be the headers of the csv file. 
        
        Example:
        log_data.add_logfile("nn_loss", ["nn_loss.csv", ['iter', 'loss', 'value_loss', 'prob_loss']])
        
        """

        self.all_charts[chart_name] = data
        self.write_headers(chart_name)

    def write_headers(self, chart_name):

        filename = self.folder + "/" + self.all_charts[chart_name][0]
        fieldnames = self.all_charts[chart_name][1]
        with open(filename, "w") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            csv_writer.writeheader()

    def save_data(self, chart_name, iter_number, data):

        filename = self.folder + "/" + self.all_charts[chart_name][0]
        fieldnames = self.all_charts[chart_name][1]
        stored_values = [iter_number] + list(data)
        with open(filename, "a") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            info = {k: v for (k, v) in zip(fieldnames, stored_values)}
            csv_writer.writerow(info)    