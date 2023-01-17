import glob
import re
import csv
import ftfy


def load_resume_filenames(dir_path=r'data/resumes_corpus/*.txt') -> list[str]:
    return [file for file in glob.glob(dir_path, recursive=True)]


def format_data(data):
    data = re.sub(r'http\S+', '', data)
    data = data.replace("Sr.", "Senior")
    data = data.replace(u'\xa0', u' ')
    data = data.replace(u'\u2022 ', u'')
    data = data.replace(". ", ".\n")
    # print(repr(data))
    data = data.split("\n")
    return list(map(str.strip, data))


def main() -> int:
    resume_filenames = load_resume_filenames()
    with open("data/job_experiences.csv", "w") as output:
        csv_writer = csv.writer(output)
        for filename in resume_filenames:
            with open(filename, encoding="ISO-8859-1") as f:
                file_data = format_data(ftfy.fix_text(f.read()))

            title = None
            job_description = []
            for line in file_data:
                if " - " in line:
                    title = line.split(" - ")[0]
                else:
                    if title:
                        for ln in re.split(r'\ \+\ |\ \*\ |\ \?\ |\ \§\ |\ \·\ ', line):
                            job_description.append([title, ln])
            if title:
                csv_writer.writerows(job_description)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
