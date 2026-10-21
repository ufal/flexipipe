<?php
/**
 * Publishing a model trained on the corpus (act=publish&folder=…, from the training page; administrators):
 * `flexipipe publish --pack --json` makes a package without the paths of this server (and with the
 * corpus's tagset description, when it has one); the page shows what was taken out, what is left to
 * look at and what the training files say about their licence, offers the package (act=publishget), and
 * opens flexipipe-hub's deposit form filled in from the model's card — the hub is another server: the
 * person signs in there, checks the form, adds the package and agrees to the terms.
 */

const FLEXIPIPE_PUBLISH_DIR = 'tmp/flexipipe-publish';

/** The package of a corpus model folder (in tmp/, for the download). */
function flexipipe_publish_package( $folder ) {
	global $foldername;
	$corpus = preg_replace( '/[^A-Za-z0-9_-]+/', '-', (string)( $foldername ?? basename( getcwd() ) ) );
	return FLEXIPIPE_PUBLISH_DIR . "/$corpus-" . preg_replace( '/[^A-Za-z0-9_-]+/', '-', basename( $folder ) ) . '.tar.gz';
}

/** The model folder named by ?folder=, when it is one of the corpus's models; else null. */
function flexipipe_publish_folder( $fp_cmd ) {
	$folder = (string)( $_GET['folder'] ?? '' );
	foreach ( flexipipe_corpus_models( $fp_cmd ) as $m ) if ( $m['folder'] === $folder ) return $m;
	return null;
}

/** act=publishget: the package, as a download (exits). */
function flexipipe_publish_download( $fp_cmd, $is_admin ) {
	$m = $is_admin ? flexipipe_publish_folder( $fp_cmd ) : null;
	$file = $m ? flexipipe_publish_package( $m['folder'] ) : '';
	if ( !$m || !is_file( $file ) ) {
		http_response_code( 404 );
		return "<p class=wrong>There is no package of this model: make it on the model's publish page.</p>";
	}
	while ( ob_get_level() ) ob_end_clean();
	header( 'Content-Type: application/gzip' );
	header( 'Content-Length: ' . filesize( $file ) );
	header( 'Content-Disposition: attachment; filename="' . basename( $file ) . '"' );
	readfile( $file );
	exit;
}

/** act=publish: make the package, show it, and the way to the deposit form. */
function flexipipe_publish_page( $fp_cmd, $action, $is_admin ) {
	$h = 'flexipipe_h';
	$out = "<h1>Publishing a model</h1><p><a href='index.php?action=$action&act=training'>Training models on this corpus</a> › publishing</p>";
	if ( !$is_admin ) return $out . "<p class=wrong>Only administrators can publish the corpus's models.</p>";
	@set_time_limit( 300 );
	$m = flexipipe_publish_folder( $fp_cmd );
	if ( !$m ) return $out . "<p class=wrong>There is no such model.</p>";
	if ( !is_dir( FLEXIPIPE_PUBLISH_DIR ) && !@mkdir( FLEXIPIPE_PUBLISH_DIR, 0775, true ) ) return $out . "<p class=wrong>The folder " . FLEXIPIPE_PUBLISH_DIR . " cannot be created.</p>";
	$package = flexipipe_publish_package( $m['folder'] );
	list( $rc, $stdout, $err ) = $fp_cmd->run( array( 'publish', '--pack', $m['folder'], '-o', $package, '--json' ) );
	$r = json_decode( trim( $stdout ), true );
	if ( !$r || empty( $r['ok'] ) ) {
		return $out . "<p class=wrong>The model could not be packed: " . $h( $r['error'] ?? trim( $err ) ?: "flexipipe exited with $rc" ) . "</p>";
	}

	$out .= "<p>The model trained " . $h( $m['created'] ?? basename( $m['folder'] ) ) . " (" . $h( $m['backend'] ?? '' ) . "): "
		. $h( flexipipe_model_scores( $m ) ) . "</p>";

	// 1 the package
	$list = function ( $rows ) use ( $h ) {
		return "<ul>" . implode( '', array_map( function ( $x ) use ( $h ) { return "<li><code>" . $h( $x[0] ) . "</code> " . $h( $x[1] ) . ": <code>" . $h( $x[2] ) . "</code></li>"; }, $rows ) ) . "</ul>";
	};
	$out .= "<h2>1. The package</h2>"
		. "<p><a class='button' href='index.php?action=$action&act=publishget&folder=" . rawurlencode( $m['folder'] ) . "'>Download the package</a> "
		. "<span class='fp-help'>" . $h( basename( $package ) ) . ", " . number_format( $r['size'] / 1048576, 1 ) . " MB"
		. ( !empty( $r['tagset'] ) ? "; with the corpus's tagset description (tagset.xml)" : '' ) . "</span></p>";
	if ( $r['cleaned'] ) $out .= "<p>Taken out of the package — paths of this server (the model here keeps them):</p>" . $list( $r['cleaned'] );
	else $out .= "<p class='fp-help'>The model names no paths of this server.</p>";
	if ( $r['look'] ) $out .= "<p class=wrong>Left in the model itself, as they may be data (word forms): look at them before depositing.</p>" . $list( $r['look'] );

	// 2 the licence
	$out .= "<h2>2. The training data's licence</h2><p class='fp-help'>The model holds material taken from the corpus's texts (for flexitag: every word form "
		. "with its frequencies and contexts): it can be published only as far as the texts may be shared.</p>"
		. "<p>What the corpus's files say: " . $h( $r['corpus_licence_text'] !== '' ? $r['corpus_licence_text'] : 'the model does not record it (trained before this was recorded)' ) . "</p>";

	// 3 the deposit form
	$out .= "<h2>3. The deposit</h2>";
	$hub = rtrim( (string)( $r['hub_url'] ?? '' ), '/' );
	if ( $hub === '' ) {
		return $out . "<p class='fp-help'>Administrators: flexipipe does not know where flexipipe-hub is — set it on the server with "
			. "<code>flexipipe config --set-model-hub-url URL</code>; the package above can be deposited by hand meanwhile.</p>";
	}
	$fields = (array)$r['fields'];
	if ( isset( $fields['steps'] ) ) $fields['steps'] = implode( ',', (array)$fields['steps'] );
	$url = "$hub/index.php?action=submit&" . http_build_query( $fields );
	$out .= "<p>The deposit form of flexipipe-hub opens filled in from the model's card. There you sign in, check and complete the description, "
		. "add the package you downloaded, and agree to the terms. Nothing is published before a curator has checked it.</p>"
		. "<p><a class='button' href='" . $h( $url ) . "' target='_blank' rel='noopener'>Open the deposit form</a></p>";
	return $out;
}
