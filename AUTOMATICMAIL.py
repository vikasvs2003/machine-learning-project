import os
import time
import psutil
import smtplib
import schedule
from sys import *
import smtplib,ssl
import urllib.error
import urllib.request
from email import encoders
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart


class ProcessLogger:
    def is_connected(self):
        try:
            urllib.request.urlopen('http://www.gmail.com')
            return True
        except urllib.error.URLError as err:
            return False

    def mail_sender(self, filename, current_time):
        try:
            fromaddr = "------@gmail.com"
            toaddr = "-------@ymail.com"

            msg = MIMEMultipart()

            msg['From'] = fromaddr
            msg['To'] = toaddr

            body = f"""
            Hello {toaddr},
            Welcome to Nikhil ML.
            Please find attached document which contains Log of Running processes.
            Log file is created at: {current_time}

            This is an auto-generated mail.

            Thanks & Regards,
            NIkhil Prakash Ahir
            NIkhil ML
            """

            subject = f"NIkhil ML Process log generated at: {current_time}"

            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))

            attachment = open(filename, "rb")

            p = MIMEBase('application', 'octet-stream')

            p.set_payload((attachment).read())

            encoders.encode_base64(p)

            p.add_header('Content-Disposition', f"attachment; filename= {filename}")
            msg.attach(p)

            s = smtplib.SMTP('smtp.gmail.com', 587)

            s.starttls()

            s.login(fromaddr, "--------")

            text = msg.as_string()

            s.sendmail(fromaddr, toaddr, text)

            s.quit()

            print("Log file successfully sent through Mail")

        except Exception as E:
            print("Unable to send mail.", E)

    def process_log(self, log_dir='NIkhil'):
        list_process = []

        if not os.path.exists(log_dir):
            try:
                os.mkdir(log_dir)
            except:
                pass

        separator = "-" * 80
        log_path = os.path.join(log_dir, f"MarvellousLog{time.ctime()}.log")
        f = open(log_path, 'w')
        f.write(separator + "\n")
        f.write("NIkhil ML Process Logger : " + time.ctime() + "\n")
        f.write(separator + "\n")
        f.write("\n")

        for proc in psutil.process_iter():
            try:
                pinfo = proc.as_dict(attrs=['pid', 'name', 'username'])
                vms = proc.memory_info().vms / (1024 * 1024)
                pinfo['vms'] = vms
                list_process.append(pinfo)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass

        for element in list_process:
            f.write(f"{element}\n")

        print(f"Log file is successfully generated at location {log_path}")

        connected = self.is_connected()

        if connected:
            start_time = time.time()
            self.mail_sender(log_path, time.ctime())
            end_time = time.time()

            print(f'Took {end_time - start_time} seconds to send mail ')
        else:
            print("There is no internet connection")


def main():
    print("---- NIkhil ML by NIkhil Ahir-----")

    print("Application name : " + argv[0])

    if len(argv) == 2:
        if argv[1] == "-h" or argv[1] == "-H":
            print("This Script is used to log records of running processes.")
            exit()

        if argv[1] == "-u" or argv[1] == "-U":
            print("Usage: ApplicationName AbsolutePath_of_Directory")
            exit()

    try:
        logger = ProcessLogger()
        schedule.every(int(argv[1])).minutes.do(logger.process_log)
        while True:
            schedule.run_pending()
            time.sleep(1)
    except ValueError:
        print("Error: Invalid datatype of input")

    except Exception as E:
        print("Error: Invalid input", E)


if __name__ == "__main__":
    main()
