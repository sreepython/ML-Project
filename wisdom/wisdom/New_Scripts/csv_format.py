import csv
import os
import re
import sys

class IssueTracker:
    def __init__(self, filename):
        self.filename = filename
        self.issues = []
        
        try:
            with open(filename, newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                
                for row in reader:
                    issue = {
                        'id': int(row['ID']),
                        'title': row['Title'],
                        'status': row['Status'],
                        'priority': row['Priority'],
                        'assignee': row['Assignee'],
                        'created_date': datetime.datetime.strptime(row['Created Date'], '%Y-%m-%d %H:%M:%S'),
                        'updated_date': datetime.datetime.strptime(row['Updated Date'], '%Y-%m-%d %H:%M:%S'),
                        'closed_date': None,
                        'comments': [],
                    }
                    
                    self.issues.append(issue)
            
        except FileNotFoundError:
            raise Exception(f'File "{filename}" does not exist.')
        except PermissionError:
            raise Exception(f'Permission denied when trying to read file "{filename}".')
        except UnicodeDecodeError:
            raise Exception(f'Unable to decode file "{filename}" using UTF-8 encoding.')
        except ValueError:
            raise Exception(f'Invalid value found while parsing file "{filename}".')
        except KeyError:
            raise Exception(f'Missing key in file "{filename}".')
        except IndexError:
            raise Exception(f'Index out of range in file "{filename}".')
        except TypeError:
            raise Exception(f'Type error occurred while processing file "{filename}".')
        except Exception as e:
            raise Exception(f'Unknown exception occurred while reading file "{filename}": {type(e)} {e}.')
    
    def get_all_issues(self):
        return self.issues
    
    def get_open_issues(self):
        return list(filter(lambda x: x['status'] == 'Open', self.issues))
    
    def get_closed_issues(self):
        return list(filter(lambda x: x['status'] == 'Closed', self.issues))
    
    def get_issue_by_id(self, id):
        matches = list(filter(lambda x: x['id'] == id, self.issues))
        if len(matches) > 0:
            return matches[0]
        else:
            return None
    
    def update_issue(self, id, updates):
        issue = self.get_issue_by_id(id)
        if issue is None:
            raise Exception(f'No issue exists with ID {id}.')
        
        for k, v in updates.items():
            issue[k] = v
        
        self._write_issues()
    
    def _write_issues(self):
        with open(self.filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['ID', 'Title', 'Status', 'Priority', 'Assignee', 'Created Date', 'Updated Date'])
            for issue in self.issues:
                writer.writerow([issue['id'], issue['title'], issue['status'], issue['priority'], issue['assignee'], issue['created_date'], issue['updated_date']])

if __name__ == '__main__':
    tracker = IssueTracker('issues.csv')
    print(tracker.get_all_issues())
    print(tracker.get_open_issues())
    print(tracker.get_closed_issues())
    print(tracker.get_issue_by_id(1))
    tracker.update_issue(1, {'status': 'Resolved'})
    print(tracker.get_issue_by_id(1))