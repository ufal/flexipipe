<?php
/**
 * Running flexipipe from TEITOK: the document path from a cid (checked), the manifest that applies
 * to a document (by key, or by `recond` on the path and `xpres` on the XML), the command line for
 * `flexipipe process` (every argument escaped), and `flexipipe info …` calls as JSON with a small
 * file cache (tmp/flexipipe-cache/) so a page view does not start Python again and again.
 */

class FlexipipeCmd
{
	var $python;
	const CACHE_DIR = 'tmp/flexipipe-cache';
	const CACHE_SECONDS = 3600;

	function __construct( $venv ) {
		$this->python = $venv->venvdir . '/bin/python';
	}

	/** The XML file of a cid (xmlfiles/…), or '' when the cid leaves xmlfiles/ or does not exist. */
	static function document_path( $cid ) {
		$cid = (string)$cid;
		if ( $cid === '' || strpos( $cid, "\0" ) !== false || strpos( $cid, '..' ) !== false || $cid[0] === '/' ) return '';
		$cid = preg_replace( '#^xmlfiles/#', '', $cid );
		if ( !preg_match( '/\.xml$/i', $cid ) ) $cid .= '.xml';
		$path = "xmlfiles/$cid";
		$real = realpath( $path );
		$base = realpath( 'xmlfiles' );
		if ( !$real || !$base || strpos( $real, $base . DIRECTORY_SEPARATOR ) !== 0 ) return '';
		return $path;
	}

	/** `python -m flexipipe ARGS…` (each argument escaped). */
	function command( $args ) {
		return escapeshellarg( $this->python ) . ' -m flexipipe ' . implode( ' ', array_map( 'escapeshellarg', $args ) );
	}

	/** Run and return [exit code, stdout, stderr]. */
	function run( $args ) {
		// stderr to a file: reading two pipes one after the other blocks as soon as flexipipe writes more
		// to stderr than a pipe holds (verbose / debug output) — both sides would wait for ever
		$errfile = tempnam( sys_get_temp_dir(), 'flexipipe-err-' );
		$spec = array( 0 => array( 'file', '/dev/null', 'r' ), 1 => array( 'pipe', 'w' ), 2 => array( 'file', $errfile, 'w' ) );
		$proc = proc_open( $this->command( $args ), $spec, $pipes );
		if ( !is_resource( $proc ) ) {
			@unlink( $errfile );
			return array( -1, '', 'flexipipe could not be started' );
		}
		$started = microtime( true );
		self::log_call( 'start', $args, $this->python );
		$out = stream_get_contents( $pipes[1] );
		fclose( $pipes[1] );
		$rc = proc_close( $proc );
		$err = (string)@file_get_contents( $errfile );
		@unlink( $errfile );
		self::log_call( sprintf( 'end %.1fs exit %d', microtime( true ) - $started, $rc ), $args );
		return array( $rc, $out, $err );
	}

	/** A line per call in tmp/flexipipe-calls.log (start, then duration and exit code; the newest
	 *  ~200 kept): what a slow or hanging page was waiting for. */
	static function log_call( $what, $args, $python = '' ) {
		$log = 'tmp/flexipipe-calls.log';
		if ( !is_dir( 'tmp' ) ) return;
		if ( @filesize( $log ) > 40000 ) {
			$lines = @file( $log );
			if ( $lines ) @file_put_contents( $log, implode( '', array_slice( $lines, -200 ) ) );
		}
		$user = function_exists( 'posix_geteuid' ) ? ( posix_getpwuid( posix_geteuid() )['name'] ?? '' ) : '';
		$new = !file_exists( $log );
		@file_put_contents( $log, date( 'Y-m-d H:i:s' ) . " [" . getmypid() . " $user] $what: " . implode( ' ', array_slice( $args, 0, 4 ) ) . ( $python !== '' ? "  ($python)" : '' ) . "\n", FILE_APPEND );
		if ( $new ) @chmod( $log, 0666 );  // the web server and admins on the command line both write it
	}

	/** `flexipipe ARGS` printing JSON: the decoded answer, null when it fails. Cached when $cache. */
	function json( $args, $cache = true ) {
		// $cache: true (an hour), false, or a number of seconds
		$seconds = $cache === true ? self::CACHE_SECONDS : (int)$cache;
		$file = self::CACHE_DIR . '/' . sha1( $this->python . "\n" . implode( "\n", $args ) ) . '.json';
		if ( $cache && file_exists( $file ) && filemtime( $file ) > time() - $seconds ) {
			$data = json_decode( file_get_contents( $file ), true );
			if ( $data !== null ) return $data;
		}
		list( $rc, $out ) = $this->run( $args );
		$data = $rc === 0 ? json_decode( trim( $out ), true ) : null;
		if ( $data !== null && $cache ) {
			if ( !is_dir( self::CACHE_DIR ) ) @mkdir( self::CACHE_DIR, 0775, true );
			@file_put_contents( $file, json_encode( $data ) );
		}
		return $data;
	}

	/** After installing or upgrading: the cached answers are out of date. */
	static function clear_cache() {
		foreach ( glob( self::CACHE_DIR . '/*' ) ?: array() as $f ) @unlink( $f );
	}

	function version() {
		$file = self::CACHE_DIR . '/version-' . sha1( $this->python ) . '.txt';
		if ( file_exists( $file ) && filemtime( $file ) > time() - self::CACHE_SECONDS ) return trim( file_get_contents( $file ) );
		list( $rc, $out ) = $this->run( array( '--version' ) );
		$v = $rc === 0 ? trim( $out ) : '';
		if ( $v !== '' ) {
			if ( !is_dir( self::CACHE_DIR ) ) @mkdir( self::CACHE_DIR, 0775, true );
			@file_put_contents( $file, $v );
		}
		return $v;
	}

	// ── manifests ────────────────────────────────────────────────────────

	static function matches_recond( $pattern, $path ) {
		if ( $pattern === '' || $pattern === null ) return true;
		return @preg_match( '#' . str_replace( '#', '\#', $pattern ) . '#u', $path ) === 1;
	}

	static function matches_xpres( $expr, $xml ) {
		if ( $expr === '' || $expr === null ) return true;
		if ( !is_string( $xml ) || $xml === '' ) return false;
		libxml_use_internal_errors( true );
		$doc = new DOMDocument();
		$ok = $doc->loadXML( $xml );
		libxml_clear_errors();
		if ( !$ok ) return false;
		$xp = new DOMXPath( $doc );
		$root = $doc->documentElement;
		if ( $root && $root->namespaceURI ) $xp->registerNamespace( 'tei', $root->namespaceURI );
		$nodes = @$xp->query( $expr );
		return $nodes && $nodes->length > 0;
	}

	/** The manifest for a document: [key, item] — by $key, else the first whose recond/xpres match. */
	static function manifest_for( $settings, $path, $key = '' ) {
		$items = $settings->profiles( 'manifests' );
		if ( $key !== '' && $key !== null ) return isset( $items[ $key ] ) ? array( $key, $items[ $key ] ) : array( null, null );
		$xml = null;
		foreach ( $items as $k => $item ) {
			if ( !self::matches_recond( $item['recond'] ?? '', $path ) ) continue;
			if ( !empty( $item['xpres'] ) ) {
				if ( $xml === null ) $xml = (string)@file_get_contents( $path );
				if ( !self::matches_xpres( $item['xpres'], $xml ) ) continue;
			}
			return array( $k, $item );
		}
		return array( null, null );
	}

	/**
	 * Arguments for `flexipipe process` on a TEITOK document.
	 * $opt: tasks (comma list; else the manifest's, else flexipipe's defaults), language, keep
	 * (attributes whose existing values stay: comma list or "all"), replace (attributes assigned
	 * anew: left out of the input),
	 * application (profile file), writeback (default true), test (preview: nothing written),
	 * verbose. Tokenization is not forced: flexipipe tokenizes only when the file has no tokens
	 * and `tokenize` is among the tasks.
	 */
	static function process_args( $path, $manifest = null, $opt = array() ) {
		$a = array( 'process', '--teitok', '--input', $path, '--api' );
		$tasks = $opt['tasks'] ?? ( $manifest['tasks'] ?? '' );
		if ( is_array( $manifest ) ) {
			if ( !empty( $manifest['backend'] ) ) array_push( $a, '--backend', $manifest['backend'] );
			if ( !empty( $manifest['model'] ) ) array_push( $a, '--model', $manifest['model'] );
			$bundle = $manifest['bundle'] ?? ( $manifest['path'] ?? '' );
			if ( $bundle !== '' && file_exists( $bundle ) ) array_push( $a, '--manifest', $bundle );
		}
		if ( $tasks !== '' ) {
			array_push( $a, '--tasks', $tasks );
			if ( in_array( 'tokenize', array_map( 'trim', explode( ',', $tasks ) ), true ) ) $a[] = '--tokenize';
		} else {
			$a[] = '--tokenize';
		}
		if ( !empty( $opt['language'] ) ) array_push( $a, '--language', $opt['language'] );
		if ( !empty( $opt['keep'] ) ) array_push( $a, '--keep', $opt['keep'] );
		if ( !empty( $opt['replace'] ) ) array_push( $a, '--replace', $opt['replace'] );
		if ( !empty( $opt['layout'] ) ) array_push( $a, '--sentence-layout', $opt['layout'] );
		if ( !empty( $opt['application'] ) && file_exists( $opt['application'] ) ) array_push( $a, '--application', $opt['application'] );
		if ( !empty( $opt['test'] ) ) $a[] = '--test';
		elseif ( !isset( $opt['writeback'] ) || $opt['writeback'] ) $a[] = '--writeback';
		if ( !empty( $opt['verbose'] ) ) $a[] = '--verbose';
		return $a;
	}
}
