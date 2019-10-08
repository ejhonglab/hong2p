/* 
 * This was copied from Google script editor accessible via
 * Tools->Script editor, from the mb_team_flies Google sheet.
 */

Array.prototype.contains = function(v) {
  for (var i = 0; i < this.length; i++) {
    if (this[i] === v) return true;
  }
  return false;
};

// https://stackoverflow.com/questions/11246758/how-to-get-unique-values-in-an-array
Array.prototype.unique = function() {
  var arr = [];
  for (var i = 0; i < this.length; i++) {
    if (!arr.contains(this[i])) {
      arr.push(this[i]);
    }
  }
  return arr;
}

// TODO maybe modify this to remove empty elements (how? possible in same line?)
Array.prototype.flat = function() {
  return [].concat.apply([], this);
}


// (both work)
/*
function test_unique() {
  var duplicates = [1, 3, 4, 2, 1, 2, 3, 8];
  var uniques = duplicates.unique();
  Logger.log(duplicates);
  Logger.log(uniques);
}
*/
/*
function test_flat() {
  var nonflat = [[1], [2], [3]];
  var nonflat_someempty = [[1], [], [2], [3], []];
  Logger.log(nonflat);
  Logger.log(nonflat_someempty);
  Logger.log(nonflat.flat());
  Logger.log(nonflat_someempty.flat());
}
*/


// For debugging. Could remove later.
// http://ramblings.mcpher.com/Home/excelquirks/gassnips/whatami
function whatAmI(ob) {
  try {
    // test for an object
    if (ob !== Object(ob)) {
        return {
          type:typeof ob,
          value: ob,
          length:typeof ob === 'string' ? ob.length : null 
        } ;
    }
    else {
      try {
        var stringify = JSON.stringify(ob);
      }
      catch (err) {
        var stringify = '{"result":"unable to stringify"}';
      }
      return {
        type:typeof ob ,
        value : stringify,
        name:ob.constructor ? ob.constructor.name : null,
        nargs:ob.constructor ? ob.constructor.arity : null,
        length:Array.isArray(ob) ? ob.length:null
      };       
    }
  }
  catch (err) {
    return {
      type:'unable to figure out what I am'
    } ;
  }
}


function getLastNonEmptyRow(sheet) {
  var range = sheet.getDataRange();
  var values = range.getValues();
  var i = values.length - 1;
  // TODO does this actually work in 0 case (if last nonempty row is 1st)?
  // (should it be i >= 0)?
  // (yea, pretty sure it won't reach 0, but that may not matter, just b/c
  // how i'm using it)
  while (i > 0) {
    var breaking = false;
    for (var j = 0; j < values[i].length; j++) {
      if (values[i][j]) {
        breaking = true;
        break;
      }
    }
    if (breaking) break;
    i--;
  }
  var last_row = i + 1;
  
  // This seemed like a more idiomatic / concise way to solve the same problem,
  // but isBlank is actually taking a while (from looking at View->Execution transcript)
  // and it's unclear if this is something I can fix, so using the above method.
  // (It was taking long enough to time out, so I effectively couldn't use this method at all)
  // ...maybe write my own custom isBlank using type of logic above?
  /*
  var last_row = sheet.getLastRow();
  while (last_row > 1) {
    var row = sheet.getRange(last_row, 1, 1, 3);
    if (! row.isBlank()) {
      break;
    }
    last_row--;
  }
  */
  
  // TODO handle/test cases where there are only/no empty rows?
  
  return last_row;
}


function onOpen() {
  var spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
  var sheet = spreadsheet.getSheetByName('recordings');
  spreadsheet.setActiveSheet(sheet);

  var last_row = getLastNonEmptyRow(sheet);
  
  // Want the next row (the first empty one).
  sheet.setCurrentCell(sheet.getRange(last_row + 1, 1));
}


// Providing optional third argument because it seems sheet.getLastColumn()
// was taking a significant portion of getSheetNamesWithCols overall run time.
function getColByName(sheet, name, last_col) {
  // Since last_col will be null if getColByName is only called with two arguments
  if (! last_col) last_col = sheet.getLastColumn();
  
  for (var i = 1; i < last_col + 1; i++) {
    if (sheet.getRange(1, i, 1, 1).getValue() === name) return i;
  }
  return null;
}


function getFlyNumCol(sheet) {
  return getColByName(sheet, 'fly_num');
}


function getSheetNamesWithCols(spreadsheet, col_names) {
  var sheet_names = [];
  var sheets = spreadsheet.getSheets();
  for (var i = 0; i < sheets.length; i++) {
    var sheet = sheets[i];
    var last_col = sheet.getLastColumn();
    var n_found_sheets = 0;
    for (var j = 0; j < col_names.length; j++) {
      if (getColByName(sheet, col_names[j], last_col)) n_found_sheets++;
    }
    if (n_found_sheets === col_names.length) sheet_names.push(sheet.getName());
  }
  return sheet_names;
}


// TODO maybe we don't always want to cast to string? some kind of optional arg to toggle?
function colContainsValue(sheet, col_num, value, last_row_to_check) {
  if (! last_row_to_check) last_row_to_check = sheet.getMaxRows();
  // Starting from row 2 because header.
  // -1 because 3rd arg is num rows, and we are starting from row 2 (1-indexed)
  var col_values = sheet.getRange(2, col_num, last_row_to_check - 1, 1).getValues();
  
  // TODO if we use this function for more than just date case test whether 
  // .flat behavior matters, maybe check emptiness of value / fix/test flat->unique
  
  // .flat may not remove empty elements, but as long as we've already returned 
  // if new_date could be empty, it shouldn't matter.
  var unique_existing_values = col_values.flat().unique().map(String);
  
  // May need to adapt how this is handled if two equal dates end up having
  // different String representations for some reason.
  // (doing it this way b/c two equal dates don't seem to compare equal
  // with either == or ===)
  return unique_existing_values.contains(String(value));
}

// TODO rather than waiting for user to enter date, maybe also add this as a 
// button / addon feature, where it will automatically enter the first date
// (today's date) in correct place, and then do everything else that gets
// triggered here
function onEdit(e) {
  var range = e.range;
  // TODO maybe also trigger on multiple-row insertions of the same date?
  // (or multi-row deletions that should trigger deletion of other things,
  // if supporting that in the single-deletion case)
  if (range.getNumRows() > 1 || range.getNumColumns() > 1) return;
  
  var spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
  var date_fly_sheet_names = getSheetNamesWithCols(spreadsheet, ['date', 'fly_num']);

  var sheet = SpreadsheetApp.getActiveSheet();
  var sheet_name = sheet.getName();
  
  var col_num = range.getColumn();
  var col_header = sheet.getRange(1, col_num, 1, 1).getValue();
  var row_num = range.getRow();
  
  if (col_header === 'date') {
    var new_date = range.getValue();
    // TODO maybe check if highlight above this row needs to be deleted?
    // (if new date was entered by mistake and this onEdit fn added highlighting)
    if (! new_date) return;
    
    // - 1 because we only want to check UP TO (but not including) the currently edited row
    if (colContainsValue(sheet, col_num, new_date, row_num - 1)) return;
    
    // The three argument version of this fn only returns a range with one column, not all.
    var row = sheet.getRange(row_num, 1, 1, sheet.getMaxColumns());
    row.setBorder(true, null, false, null, false, false, 'black', SpreadsheetApp.BorderStyle.SOLID_THICK);
    
    if (! date_fly_sheet_names.contains(sheet_name)) return;
    
    var fly_num_cell = sheet.getRange(row_num, getFlyNumCol(sheet), 1, 1);
    if (! fly_num_cell.getValue()) fly_num_cell.setValue(1);
    
    for (var i = 0; i < date_fly_sheet_names.length; i++) {
      var sn = date_fly_sheet_names[i];
      if (sn === sheet_name) continue;
      
      var other_sheet = spreadsheet.getSheetByName(sn);
      var last_row = getLastNonEmptyRow(other_sheet) + 1;
      col_num = getColByName(other_sheet, 'date');
      // could mayyybe still add missing fly_num = 1 / format in this case
      if (colContainsValue(other_sheet, col_num, new_date, last_row - 1)) continue;
      
      // TODO maybe just change all date cols to correct number format / data validation in one place
      // (like on load)?
      other_sheet.getRange(last_row, col_num, 1, 1).setValue(new_date).setNumberFormat('yyyy-mm-dd');
      
      row = other_sheet.getRange(last_row, 1, 1, other_sheet.getMaxColumns());
      row.setBorder(true, null, false, null, false, false, 'black', SpreadsheetApp.BorderStyle.SOLID_THICK);
      
      // TODO maybe the multiple getColByName calls here mean i should just modify date_fly_sheet_names
      // into something that also points to where they are, so they are only computed once?
      other_sheet.getRange(last_row, getColByName(other_sheet, 'fly_num'), 1, 1).setValue(1);
    }
    
  } else if (col_header === 'fly_num') {
    var fly_num = range.getValue();
    if (! fly_num || fly_num === 1) return;
    
    var last_row = getLastNonEmptyRow(sheet);
    if (row_num !== last_row) return;
    
    var i = row_num;
    var last_fly_num = fly_num;
    var todays_date = null;
    var date_col = getColByName(sheet, 'date');
    while (i >= 2) {
      var curr_fly_num = sheet.getRange(i, col_num, 1, 1).getValue();
      // assert this var is an int? / cast?
      
      // could maybe support this case, but not sure it matters
      if (! curr_fly_num) {
        Logger.log('missing fly_num when looking for date');
        return;
      }
      
      // Could also restrict this to always needing to decrease by one, but i
      // don't think i want that. Might skip bad flies.
      if (curr_fly_num > last_fly_num) {
        if (!todays_date) Logger.log('increasing fly_num before date');
        break;
      }
      // Checking any dates encountered along the way to the last 1 are consistent.
      var curr_date = String(sheet.getRange(i, date_col, 1, 1).getValue());
      if (curr_date) {
        if (! todays_date) {
          todays_date = curr_date;
        } else if (todays_date !== curr_date) {
          Logger.log('inconsistent dates');
          return;
        }
      }
      last_fly_num = curr_fly_num;
      i--;
    }
    if (! todays_date) {
      Logger.log('no date found');
      return;
    }
    // only supporting this case to be conservative about when fly_nums are screwed with across sheets
    Logger.log('found date ' + todays_date);
    
    for (var i = 0; i < date_fly_sheet_names.length; i++) {
      var sn = date_fly_sheet_names[i];
      if (sn === sheet_name) continue;
      
      var other_sheet = spreadsheet.getSheetByName(sn);
      var fly_col = getColByName(other_sheet, 'fly_num');
      var last_row = getLastNonEmptyRow(other_sheet) + 1;
      
      // This may or may not be overly restrictive. if using other onEvent trigger
      // s.t. fly_num == 1 is always set, it should probably be fine.
      prior_fly = fly_num - 1;
      if (other_sheet.getRange(last_row - 1, fly_col, 1, 1).getValue() !== prior_fly) {
        Logger.log('sheet ' + sn + ' did not have preceding fly_num = ' + String(prior_fly));
        continue;
      }
      // may want to also assert correct date in fly_num progression on other sheet,
      // but that may be overkill
      other_sheet.getRange(last_row, fly_col, 1, 1).setValue(fly_num);
    }
  }
}

// TODO TODO could have certain columns only show if it's signed in under my email,
// otherwise hide all the rows that remy hides, for her sake
// https://developers.google.com/apps-script/reference/base/session

// TODO possible to have a trigger that changes to the bottom of the new tab whenever the
// user switches tabs?

// TODO apply my data validation to the whole of each column, including as the spreadsheet
// grows with new chunks of blank rows added at the bottom

// TODO data validate increasing fly_num within date?
// + unique thorimage / thorsync w/in date?


