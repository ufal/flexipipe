<?php
/**
 * Background jobs for flexipipe runs that take longer than a page request (training a model on the
 * corpus; later: processing many documents). Each job is a folder in tmp/flexipipe-jobs/<id>/:
 *
 *   job.json   what was started: kind, label, arguments, who, when, the process id
 *   log        everything flexipipe printed (stdout and stderr)
 *   exit       its exit code, written when it ends (no file: still running, or gone)
 *   cancelled  present when it was stopped from the page
 *
 * A job runs detached (in the background, at lower priority, hang-ups ignored) from the corpus folder, so the page returns at once
 * and stays usable; one job per kind runs at a time (a lock around the check and the start). Its state
 * comes from those files and whether the process is still there; progress from flexipipe's
 * "[flexipipe] step 2/5: …" lines (flexipipe/progress.py). The newest KEEP jobs are kept.
 * Included by flexipipe.php ($fp_cmd).
 */

class FlexipipeJobs
{
	const DIR = 'tmp/flexipipe-jobs';
	const KEEP = 30;

	var $cmd;

	function __construct( $fp_cmd ) {
		$this->cmd = $fp_cmd;
	}

	static function valid_id( $id ) {
		return is_string( $id ) && preg_match( '/^\d{8}-\d{6}-[0-9a-f]{6}$/', $id );
	}

	function dir( $id ) {
		return self::DIR . "/$id";
	}

	/** Start `flexipipe ARGS` in the background. Returns [id, ''] or ['', the reason it did not start]. */
	function start( $kind, $label, $args, $who = '' ) {
		if ( !is_dir( self::DIR ) && !@mkdir( self::DIR, 0755, true ) ) return array( '', 'the folder for background jobs (' . self::DIR . ') cannot be created' );
		$lock = fopen( self::DIR . '/.lock', 'c' );
		if ( !$lock || !flock( $lock, LOCK_EX ) ) return array( '', 'the job list is locked' );
		try {
			foreach ( $this->all() as $job ) {
				if ( $job['kind'] === $kind && $job['status'] === 'running' ) {
					return array( '', "another job of this kind is still running (started " . $job['started'] . ")" );
				}
			}
			$id = date( 'Ymd-His' ) . '-' . bin2hex( random_bytes( 3 ) );
			$dir = $this->dir( $id );
			if ( !@mkdir( $dir, 0755 ) ) return array( '', "the job folder $dir cannot be created" );
			$job = array( 'id' => $id, 'kind' => $kind, 'label' => $label, 'args' => array_values( $args ), 'who' => $who, 'started' => date( 'Y-m-d H:i:s' ) );
			file_put_contents( "$dir/job.json", json_encode( $job, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE ) );
			$root = realpath( '.' );
			$log = escapeshellarg( realpath( $dir ) . '/log' );
			$exit = escapeshellarg( realpath( $dir ) . '/exit' );
			// the job: from the corpus folder, at lower priority; its exit code is written when it ends
			// no nohup: under Apache it cannot detach from a console and stops without running anything;
			// the shell ignores the hang-up signal itself, every stream is redirected
			$inner = "trap '' HUP; cd " . escapeshellarg( $root ) . ' && nice -n 10 ' . $this->cmd->command( $args ) . " > $log 2>&1; echo \$? > $exit";
			$pid = trim( (string)shell_exec( 'sh -c ' . escapeshellarg( $inner ) . ' > /dev/null 2>&1 < /dev/null & echo $!' ) );
			if ( !ctype_digit( $pid ) ) {
				file_put_contents( "$dir/exit", "-1\n" );
				return array( '', 'the job could not be started' );
			}
			$job['pid'] = (int)$pid;
			file_put_contents( "$dir/job.json", json_encode( $job, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE ) );
			return array( $id, '' );
		} finally {
			flock( $lock, LOCK_UN );
			fclose( $lock );
		}
	}

	static function alive( $pid ) {
		if ( !$pid ) return false;
		if ( function_exists( 'posix_kill' ) ) return posix_kill( (int)$pid, 0 );
		return trim( (string)shell_exec( 'ps -p ' . (int)$pid . ' -o pid= 2>/dev/null' ) ) !== '';
	}

	/** A job and its state: status running | finished | failed | cancelled | lost, progress, times. */
	function get( $id ) {
		if ( !self::valid_id( $id ) ) return null;
		$dir = $this->dir( $id );
		$job = json_decode( (string)@file_get_contents( "$dir/job.json" ), true );
		if ( !is_array( $job ) ) return null;
		$log = (string)@file_get_contents( "$dir/log" );
		if ( file_exists( "$dir/cancelled" ) ) {
			$job['status'] = 'cancelled';
			$job['ended'] = date( 'Y-m-d H:i:s', filemtime( "$dir/cancelled" ) );
		} elseif ( file_exists( "$dir/exit" ) ) {
			$job['exit'] = (int)trim( file_get_contents( "$dir/exit" ) );
			$job['status'] = $job['exit'] === 0 ? 'finished' : 'failed';
			$job['ended'] = date( 'Y-m-d H:i:s', filemtime( "$dir/exit" ) );
		} else {
			// no exit code: running, or gone without finishing (the machine restarted, it was killed)
			$job['status'] = self::alive( $job['pid'] ?? 0 ) ? 'running' : 'lost';
		}
		$job['progress'] = null;
		if ( preg_match_all( '/^\[flexipipe\] step (\d+)\/(\d+): (.*)$/m', $log, $m, PREG_SET_ORDER ) ) {
			$last = end( $m );
			$job['progress'] = array( 'step' => (int)$last[1], 'steps' => (int)$last[2], 'text' => trim( $last[3] ) );
		}
		$job['problem'] = $job['status'] === 'failed' ? self::problem( $log ) : '';
		$job['seconds'] = strtotime( $job['ended'] ?? date( 'Y-m-d H:i:s' ) ) - strtotime( $job['started'] );
		return $job;
	}

	/** The line that says what went wrong: the last error line, else the last line (warnings left out). */
	static function problem( $log ) {
		$lines = array_values( array_filter( array_map( 'trim', explode( "\n", $log ) ), function ( $l ) {
			return $l !== '' && stripos( $l, 'warning' ) === false;
		} ) );
		for ( $i = count( $lines ) - 1; $i >= 0; $i-- ) {
			if ( preg_match( '/(Error|error:|failed|cannot|does not exist|not found|No such)/', $lines[ $i ] ) ) return $lines[ $i ];
		}
		return $lines ? end( $lines ) : '';
	}

	function log_tail( $id, $lines = 40 ) {
		if ( !self::valid_id( $id ) ) return '';
		$all = explode( "\n", rtrim( (string)@file_get_contents( $this->dir( $id ) . '/log' ) ) );
		return implode( "\n", array_slice( $all, -$lines ) );
	}

	/** All jobs, newest first; the oldest beyond KEEP that are not running are removed. */
	function all() {
		$ids = array();
		foreach ( (array)@scandir( self::DIR ) as $name ) if ( self::valid_id( $name ) ) $ids[] = $name;
		rsort( $ids );
		$jobs = array();
		foreach ( $ids as $n => $id ) {
			$job = $this->get( $id );
			if ( !$job ) continue;
			if ( $n >= self::KEEP && $job['status'] !== 'running' ) {
				$this->remove( $id );
				continue;
			}
			$jobs[] = $job;
		}
		return $jobs;
	}

	function remove( $id ) {
		if ( !self::valid_id( $id ) ) return;
		$dir = $this->dir( $id );
		foreach ( (array)@scandir( $dir ) as $f ) if ( $f !== '.' && $f !== '..' ) @unlink( "$dir/$f" );
		@rmdir( $dir );
	}

	/** Stop a running job (flexipipe and the shell around it). */
	function cancel( $id ) {
		$job = $this->get( $id );
		if ( !$job || $job['status'] !== 'running' ) return false;
		$pid = (int)$job['pid'];
		shell_exec( "pkill -TERM -P $pid 2>/dev/null; kill -TERM $pid 2>/dev/null" );
		file_put_contents( $this->dir( $id ) . '/cancelled', date( 'c' ) . "\n" );
		return true;
	}
}

function flexipipe_duration( $seconds ) {
	$s = max( 0, (int)$seconds );
	if ( $s < 60 ) return "$s s";
	if ( $s < 3600 ) return floor( $s / 60 ) . " min";
	return floor( $s / 3600 ) . " h " . floor( ( $s % 3600 ) / 60 ) . " min";
}

/** A job's state in words: "running — step 2/5: building the vocabulary (4 min)". */
function flexipipe_job_state( $job ) {
	$h = 'flexipipe_h';
	$p = $job['progress'];
	$step = $p ? "step {$p['step']}/{$p['steps']}: " . $h( $p['text'] ) : '';
	switch ( $job['status'] ) {
		case 'running': return "running" . ( $step ? " — $step" : '' ) . " (" . flexipipe_duration( $job['seconds'] ) . ")";
		case 'finished': return "<span class=right>finished</span> after " . flexipipe_duration( $job['seconds'] );
		case 'failed': return "<span class=wrong>failed</span> after " . flexipipe_duration( $job['seconds'] ) . ( $job['problem'] ? ": " . $h( $job['problem'] ) : '' );
		case 'cancelled': return "stopped" . ( $step ? " at $step" : '' );
		default: return "<span class=wrong>ended without finishing</span>" . ( $step ? " at $step" : '' ) . " (the process is gone: the server may have restarted)";
	}
}

/** ?api=job&job=ID: the state as JSON (for the job page while it runs). */
function flexipipe_jobs_api( $jobs, $is_admin ) {
	$job = $jobs->get( (string)( $_GET['job'] ?? '' ) );
	$out = $job ? array( 'ok' => true, 'status' => $job['status'], 'state' => flexipipe_job_state( $job ), 'progress' => $job['progress'] ) : array( 'ok' => false );
	if ( $job && $is_admin ) $out['log'] = $jobs->log_tail( $job['id'] );
	header( 'Content-Type: application/json' );
	echo json_encode( $out );
	exit;
}

/** One job: state, progress, times, (admins) the stop button and the last lines of the log; while it
 *  runs, followed every few seconds (?api=job), the page reloaded when it ends. */
function flexipipe_job_panel( $jobs, $job, $action, $is_admin ) {
	$h = 'flexipipe_h';
	$id = $job['id'];
	$p = $job['progress'];
	$bar = $p ? "<progress max='" . (int)$p['steps'] . "' value='" . ( $job['status'] === 'finished' ? (int)$p['steps'] : max( 0, (int)$p['step'] - 1 ) ) . "'></progress>" : '';
	$out = "<table class='fp-table'>"
		. "<tr><th>State<td id='fp-job-state'>" . flexipipe_job_state( $job ) . "</tr>"
		. "<tr><th>Progress<td id='fp-job-bar'>$bar</tr>"
		. "<tr><th>Started<td>" . $h( $job['started'] ) . ( $job['who'] ? " by " . $h( $job['who'] ) : '' ) . "</tr>"
		. ( isset( $job['ended'] ) ? "<tr><th>Ended<td>" . $h( $job['ended'] ) . "</tr>" : '' )
		. "</table>";
	if ( $is_admin && $job['status'] === 'running' ) {
		$out .= "<form method='post' action='index.php?action=$action&act=jobcancel&job=" . rawurlencode( $id ) . "' onsubmit='return confirm(\"Stop this job?\")'><p><button type='submit'>Stop the job</button></p></form>";
	}
	if ( $is_admin ) {
		$out .= "<details" . ( $job['status'] === 'running' || $job['status'] === 'failed' ? ' open' : '' ) . "><summary>flexipipe's output (the last lines)</summary><pre id='fp-job-log'>" . $h( $jobs->log_tail( $id ) ) . "</pre>"
			. "<p class='fp-help'>flexipipe " . $h( implode( ' ', $job['args'] ) ) . "</p></details>";
	}
	if ( $job['status'] === 'running' ) {
		$api = json_encode( "index.php?action=$action&api=job&job=" . rawurlencode( $id ) );
		$out .= "<script>(function(){var t=setInterval(function(){fetch($api,{credentials:'same-origin'}).then(function(r){return r.json();}).then(function(d){if(!d.ok)return;"
			. "document.getElementById('fp-job-state').innerHTML=d.state;var l=document.getElementById('fp-job-log');if(l&&d.log!==undefined)l.textContent=d.log;"
			. "var b=document.querySelector('#fp-job-bar progress');if(b&&d.progress){b.max=d.progress.steps;b.value=Math.max(0,d.progress.step-1);}"
			. "if(d.status!=='running'){clearInterval(t);location.reload();}}).catch(function(){});},4000);})();</script>";
	}
	return $out;
}

/** act=jobs (the list), act=job&job=ID (one job), act=jobcancel (POST). */
function flexipipe_jobs_page( $jobs, $action, $act, $is_admin ) {
	$h = 'flexipipe_h';
	$out = "<h1>flexipipe: background jobs</h1>";
	$id = (string)( $_GET['job'] ?? '' );
	if ( $act === 'jobcancel' && $_SERVER['REQUEST_METHOD'] === 'POST' ) {
		if ( !$is_admin ) return $out . "<p class=wrong>Only administrators can stop a job.</p>";
		$out .= $jobs->cancel( $id ) ? "<p>The job was stopped.</p>" : "<p class=wrong>The job was not running.</p>";
		$act = 'job';
	}
	if ( $act === 'job' ) {
		$job = $jobs->get( $id );
		if ( !$job ) return $out . "<p class=wrong>There is no such job.</p><p><a href='index.php?action=$action&act=jobs'>All jobs</a></p>";
		$out .= "<h2>" . $h( $job['label'] ) . "</h2>" . flexipipe_job_panel( $jobs, $job, $action, $is_admin );
		return $out . "<p><a href='index.php?action=$action&act=jobs'>All jobs</a> · <a href='index.php?action=$action'>flexipipe</a></p>";
	}
	// the list
	$all = $jobs->all();
	if ( !$all ) return $out . "<p>No jobs have run in this corpus.</p><p><a href='index.php?action=$action'>flexipipe</a></p>";
	$out .= "<table class='fp-table'><thead><tr><th>Started<th>Job<th>By<th>State</tr></thead><tbody>";
	foreach ( $all as $job ) {
		$out .= "<tr><td>" . $h( $job['started'] ) . "<td><a href='index.php?action=$action&act=job&job=" . rawurlencode( $job['id'] ) . "'>" . $h( $job['label'] ) . "</a><td>" . $h( $job['who'] ) . "<td>" . flexipipe_job_state( $job ) . "</tr>";
	}
	return $out . "</tbody></table><p><a href='index.php?action=$action'>flexipipe</a></p>";
}
