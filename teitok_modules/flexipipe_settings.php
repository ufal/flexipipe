<?php
/**
 * flexipipe's settings in a TEITOK project (Resources/settings.xml), read and written in one place,
 * the way flexipipe itself reads them (flexipipe/teitok_settings.py):
 *
 *   <defaults lang="xx">                      default language (empty: detected)
 *     <flexipipe backend="…" model="…"        default backend / model
 *                download-model="yes|no" known-tags-only="yes|no" use-raw-text="yes|no">
 *       <models><item key="xx" backend="…" model="…"/></models>    per-language models
 *       <datasets|applications|manifests><item key="…" …/></…>       profile registries
 *       <tasks><item key="ner" backend="nametag" model="…"/></tasks>   the backend for a step the main one does not do
 *       <layers><item key="xpos" attribute="xpos"/></layers>   where a layer is written (else: automatic, the corpus's name)
 *       <training tags="tag" lemma="lemma" pron="phon" pron-context="next" use="Resources/flexipipe-models/…"/>
 *                                       training a model on the corpus, and the one in use (flexipipe_train.php)
 *       <segmentation preset="conversation"><elements><item key="seg[type=pw]" name="seg" type="pw" crossing="wrap"/></elements></segmentation>
 *                                       how sentences relate to u / seg / p (flexipipe's sentence_layout)
 *     </flexipipe>
 *   </defaults>
 *
 * Lists are TEITOK's keyed lists (<…><item key>), which xmlflatten turns into a hash; the older repeated
 * elements (<task>, <layer>, <element>) are read too and replaced when written.
 * flexipipe reads the first <flexipipe> element in the file; so does this class (and it writes
 * there, creating <defaults><flexipipe> when there is none). Per-language models are <models><item
 * key>, not <language> (several <language> elements overwrite each other in TEITOK's getset). Flags
 * are written hyphenated; the underscore spellings of older versions are read and replaced.
 * Saving keeps a daily backup (backups/settings-YYYYMMDD.xml), checks the XML and replaces the
 * file atomically.
 */

class FlexipipeSettings
{
	const FLAGS = array( 'download-model', 'known-tags-only', 'use-raw-text' );
	const GROUPS = array( 'datasets', 'applications', 'manifests' );
	// steps another backend than the main one can do (flexipipe's task_routing.SECONDARY)
	const TASKS = array( 'ner' => array( 'nametag' ), 'normalize' => array( 'hunspell' ), 'pron' => array( 'light', 'phonetisaurus' ) );

	var $path;
	var $doc;
	var $error = '';

	function __construct( $path = 'Resources/settings.xml' ) {
		$this->path = $path;
		$this->doc = new DOMDocument();
		$this->doc->preserveWhiteSpace = true;
		libxml_use_internal_errors( true );
		if ( !file_exists( $path ) || !$this->doc->load( $path ) ) {
			$this->error = "$path could not be read";
			$this->doc->loadXML( '<ttsettings/>' );
		}
		libxml_clear_errors();
	}

	/** The <flexipipe> element flexipipe reads (the first one), or null. */
	function section( $create = false ) {
		$xp = new DOMXPath( $this->doc );
		$found = $xp->query( '(//flexipipe)[1]' );
		if ( $found->length ) return $found->item( 0 );
		if ( !$create ) return null;
		return $this->defaults( true )->appendChild( $this->doc->createElement( 'flexipipe' ) );
	}

	function defaults( $create = false ) {
		$root = $this->doc->documentElement;
		foreach ( $root->childNodes as $n ) {
			if ( $n->nodeType === XML_ELEMENT_NODE && $n->nodeName === 'defaults' ) return $n;
		}
		return $create ? $root->appendChild( $this->doc->createElement( 'defaults' ) ) : null;
	}

	function language() {
		$d = $this->defaults();
		if ( !$d ) return '';
		return $d->getAttribute( 'lang' ) ?: $d->getAttribute( 'language' );
	}

	function get( $name ) {
		$s = $this->section();
		if ( !$s ) return '';
		if ( $s->hasAttribute( $name ) ) return $s->getAttribute( $name );
		$old = str_replace( '-', '_', $name );
		return $s->hasAttribute( $old ) ? $s->getAttribute( $old ) : '';
	}

	function flag( $name ) {
		return in_array( strtolower( $this->get( $name ) ), array( 'yes', 'true', '1', 'on' ), true );
	}

	/** Per-language models: [ code => [backend, model] ] (also reads older <language code> entries). */
	function language_models() {
		$out = array();
		$s = $this->section();
		if ( !$s ) return $out;
		foreach ( $s->childNodes as $n ) {
			if ( $n->nodeType !== XML_ELEMENT_NODE ) continue;
			if ( $n->nodeName === 'language' && $n->getAttribute( 'code' ) !== '' ) {
				$out[ $n->getAttribute( 'code' ) ] = array( 'backend' => $n->getAttribute( 'backend' ), 'model' => $n->getAttribute( 'model' ) );
			}
			if ( $n->nodeName === 'models' ) {
				foreach ( $n->childNodes as $it ) {
					if ( $it->nodeType === XML_ELEMENT_NODE && $it->nodeName === 'item' && $it->getAttribute( 'key' ) !== '' ) {
						$out[ $it->getAttribute( 'key' ) ] = array( 'backend' => $it->getAttribute( 'backend' ), 'model' => $it->getAttribute( 'model' ) );
					}
				}
			}
		}
		return $out;
	}

	/** A profile registry (datasets / applications / manifests): [ key => [attr => value] ], in file order. */
	function profiles( $group ) {
		$out = array();
		$s = $this->section();
		if ( !$s || !in_array( $group, self::GROUPS, true ) ) return $out;
		foreach ( $s->childNodes as $g ) {
			if ( $g->nodeType !== XML_ELEMENT_NODE || $g->nodeName !== $group ) continue;
			foreach ( $g->childNodes as $it ) {
				if ( $it->nodeType !== XML_ELEMENT_NODE || $it->nodeName !== 'item' ) continue;
				$attrs = array();
				foreach ( $it->attributes as $a ) $attrs[ $a->name ] = $a->value;
				$key = $attrs['key'] ?? (string)( count( $out ) + 1 );
				$out[ $key ] = $attrs;
			}
		}
		return $out;
	}

	/** Set the defaults: language, backend, model, flags, per-language models ([code => [backend, model]]). */
	function set_defaults( $language, $backend, $model, $flags, $per_language ) {
		$this->defaults( true )->setAttribute( 'lang', $language );
		$s = $this->section( true );
		$s->setAttribute( 'backend', $backend );
		$s->setAttribute( 'model', $model );
		foreach ( self::FLAGS as $f ) {
			$s->removeAttribute( str_replace( '-', '_', $f ) );
			$s->setAttribute( $f, !empty( $flags[ $f ] ) ? 'yes' : 'no' );
		}
		// per-language models: one <models> (older <language> entries are replaced by it)
		foreach ( iterator_to_array( $s->childNodes ) as $n ) {
			if ( $n->nodeType === XML_ELEMENT_NODE && ( $n->nodeName === 'language' || $n->nodeName === 'models' ) ) $s->removeChild( $n );
		}
		if ( $per_language ) {
			$models = $s->appendChild( $this->doc->createElement( 'models' ) );
			foreach ( $per_language as $code => $bm ) {
				$it = $models->appendChild( $this->doc->createElement( 'item' ) );
				$it->setAttribute( 'key', $code );
				$it->setAttribute( 'backend', $bm['backend'] ?? '' );
				$it->setAttribute( 'model', $bm['model'] ?? '' );
			}
		}
	}

	/** The corpus's sentence layout preset (<flexipipe><segmentation preset>; '' = flexipipe's default, written). */
	function layout_preset() {
		$s = $this->section();
		if ( !$s ) return '';
		foreach ( $s->childNodes as $n ) {
			if ( $n->nodeType === XML_ELEMENT_NODE && $n->nodeName === 'segmentation' ) return $n->getAttribute( 'preset' );
		}
		return '';
	}

	/** Set it (its <elements> settings stay); '' removes the preset. */
	function set_layout_preset( $preset ) {
		$s = $this->section( true );
		$seg = null;
		foreach ( $s->childNodes as $n ) {
			if ( $n->nodeType === XML_ELEMENT_NODE && $n->nodeName === 'segmentation' ) $seg = $n;
		}
		if ( $preset === '' ) {
			if ( $seg ) $seg->removeAttribute( 'preset' );
			return;
		}
		if ( !$seg ) $seg = $s->appendChild( $this->doc->createElement( 'segmentation' ) );
		$seg->setAttribute( 'preset', $preset );
	}

	/** A keyed list in TEITOK's form (<LIST><item key="…" …/></LIST>: xmlflatten makes it a hash) as
	 *  [key => [attr => value]]; the older repeated <OLD name="…"> elements are read too. */
	function keyed_items( $list, $old ) {
		$out = array();
		$s = $this->section();
		if ( !$s ) return $out;
		foreach ( $s->childNodes as $n ) {
			if ( $n->nodeType !== XML_ELEMENT_NODE ) continue;
			$items = array();
			if ( $n->nodeName === $old ) $items[] = $n;
			if ( $n->nodeName === $list ) {
				foreach ( $n->childNodes as $it ) if ( $it->nodeType === XML_ELEMENT_NODE && $it->nodeName === 'item' ) $items[] = $it;
			}
			foreach ( $items as $it ) {
				$key = $it->getAttribute( 'key' ) !== '' ? $it->getAttribute( 'key' ) : $it->getAttribute( 'name' );
				if ( $key === '' ) continue;
				$attrs = array();
				foreach ( $it->attributes as $a ) if ( $a->name !== 'key' && $a->name !== 'name' ) $attrs[ $a->name ] = $a->value;
				$out[ $key ] = $attrs;
			}
		}
		return $out;
	}

	/** Write a keyed list ([key => [attr => value]]; empty: the list goes), replacing it and the older form. */
	function set_keyed_items( $list, $old, $items ) {
		$s = $this->section( true );
		foreach ( iterator_to_array( $s->childNodes ) as $n ) {
			if ( $n->nodeType === XML_ELEMENT_NODE && ( $n->nodeName === $list || $n->nodeName === $old ) ) $s->removeChild( $n );
		}
		if ( !$items ) return;
		$el = $s->appendChild( $this->doc->createElement( $list ) );
		foreach ( $items as $key => $attrs ) {
			$it = $el->appendChild( $this->doc->createElement( 'item' ) );
			$it->setAttribute( 'key', $key );
			foreach ( $attrs as $k => $v ) if ( (string)$v !== '' ) $it->setAttribute( $k, $v );
		}
	}

	/** The backend per step the main backend may not do (<tasks><item key backend model>, and EasyCorp's
	 *  ner_backend / normalize_backend / pron_backend attributes; <tasks> wins): [ task => [backend, model] ]. */
	function task_routes() {
		$out = array();
		$s = $this->section();
		if ( !$s ) return $out;
		foreach ( array_keys( self::TASKS ) as $t ) {
			if ( $s->getAttribute( "{$t}_backend" ) !== '' ) $out[ $t ] = array( 'backend' => $s->getAttribute( "{$t}_backend" ), 'model' => $s->getAttribute( "{$t}_model" ) );
		}
		foreach ( $this->keyed_items( 'tasks', 'task' ) as $t => $a ) {
			if ( ( $a['backend'] ?? '' ) !== '' ) $out[ $t ] = array( 'backend' => $a['backend'], 'model' => $a['model'] ?? '' );
		}
		return $out;
	}

	/** Set them ([ task => [backend, model] ]; a task left out or with no backend: none). EasyCorp's
	 *  attributes, where the corpus has them, are kept in step (EasyCorp runs its passes from them). */
	function set_task_routes( $routes ) {
		$s = $this->section( true );
		$items = array();
		foreach ( array_keys( self::TASKS ) as $t ) {
			$r = $routes[ $t ] ?? array();
			$backend = trim( (string)( $r['backend'] ?? '' ) );
			$model = trim( (string)( $r['model'] ?? '' ) );
			if ( $s->hasAttribute( "{$t}_backend" ) ) {
				if ( $backend === '' ) {
					$s->removeAttribute( "{$t}_backend" );
					$s->removeAttribute( "{$t}_model" );
				} else {
					$s->setAttribute( "{$t}_backend", $backend );
					$s->setAttribute( "{$t}_model", $model );
				}
			}
			if ( $backend !== '' ) $items[ $t ] = array( 'backend' => $backend, 'model' => $model );
		}
		$this->set_keyed_items( 'tasks', 'task', $items );
	}

	/** Training on the corpus (<flexipipe><training tags lemma pron pron-context use>): [attr => value]. */
	function training() {
		$out = array();
		$s = $this->section();
		if ( !$s ) return $out;
		foreach ( $s->childNodes as $n ) {
			if ( $n->nodeType === XML_ELEMENT_NODE && $n->nodeName === 'training' ) {
				foreach ( $n->attributes as $a ) $out[ $a->name ] = $a->value;
			}
		}
		return $out;
	}

	/** Set training attributes ('' removes one); the others stay. */
	function set_training( $attrs ) {
		$s = $this->section( true );
		$el = null;
		foreach ( $s->childNodes as $n ) {
			if ( $n->nodeType === XML_ELEMENT_NODE && $n->nodeName === 'training' ) $el = $n;
		}
		if ( !$el ) $el = $s->appendChild( $this->doc->createElement( 'training' ) );
		foreach ( $attrs as $k => $v ) {
			if ( (string)$v === '' ) $el->removeAttribute( $k );
			else $el->setAttribute( $k, (string)$v );
		}
	}

	/** The corpus's token attributes (<cqp><pattributes><item key display type>): [key => [display, type]]. */
	function token_attributes() {
		$out = array();
		$xp = new DOMXPath( $this->doc );
		foreach ( $xp->query( '//cqp/pattributes/item[@key]' ) as $it ) {
			$out[ $it->getAttribute( 'key' ) ] = array( 'display' => $it->getAttribute( 'display' ) ?: $it->getAttribute( 'key' ), 'type' => $it->getAttribute( 'type' ) );
		}
		return $out;
	}

	/** Where the user has flexipipe write a layer (<layers><item key attribute>): [layer => attribute]. */
	function layer_attributes() {
		$out = array();
		foreach ( $this->keyed_items( 'layers', 'layer' ) as $layer => $a ) {
			if ( in_array( strtolower( $a['exclude'] ?? '' ), array( 'yes', 'true', '1' ), true ) ) { $out[ $layer ] = '-'; continue; }
			$att = $a['attribute'] ?? ( $a['att'] ?? '' );
			if ( $att !== '' ) $out[ $layer ] = $att;
		}
		return $out;
	}

	/** Set them ([layer => attribute]; '' or left out: automatic; '-': left out, exclude="yes"). */
	function set_layer_attributes( $map ) {
		$items = array();
		foreach ( $map as $layer => $attribute ) {
			if ( (string)$attribute === '-' ) $items[ $layer ] = array( 'exclude' => 'yes' );
			elseif ( (string)$attribute !== '' ) $items[ $layer ] = array( 'attribute' => $attribute );
		}
		$this->set_keyed_items( 'layers', 'layer', $items );
	}

	/** The corpus's main model (<flexipipe backend model>). */
	function set_main_model( $backend, $model ) {
		$s = $this->section( true );
		$s->setAttribute( 'backend', $backend );
		$s->setAttribute( 'model', $model );
	}

	/** The model for one language (<models><item key backend model>), added or replaced. */
	function set_language_model( $code, $backend, $model ) {
		$s = $this->section( true );
		$models = null;
		foreach ( $s->childNodes as $n ) {
			if ( $n->nodeType === XML_ELEMENT_NODE && $n->nodeName === 'models' ) $models = $n;
		}
		if ( !$models ) $models = $s->appendChild( $this->doc->createElement( 'models' ) );
		$item = null;
		foreach ( $models->childNodes as $it ) {
			if ( $it->nodeType === XML_ELEMENT_NODE && $it->nodeName === 'item' && $it->getAttribute( 'key' ) === $code ) $item = $it;
		}
		if ( !$item ) {
			$item = $models->appendChild( $this->doc->createElement( 'item' ) );
			$item->setAttribute( 'key', $code );
		}
		$item->setAttribute( 'backend', $backend );
		$item->setAttribute( 'model', $model );
	}

	/** Where a model folder is the corpus's model (the default model, a per-language model), put another. */
	function replace_model( $old, $new ) {
		$s = $this->section();
		if ( !$s ) return;
		if ( $s->getAttribute( 'model' ) === $old ) $s->setAttribute( 'model', $new );
		$xp = new DOMXPath( $this->doc );
		foreach ( $xp->query( './/models/item | .//language', $s ) as $it ) {
			if ( $it->getAttribute( 'model' ) === $old ) $it->setAttribute( 'model', $new );
		}
	}

	/** Write the file: daily backup, XML checked, atomic replace. Returns '' or the problem. */
	function save() {
		if ( $this->error ) return $this->error;
		if ( !is_writable( $this->path ) ) return "{$this->path} is not writable";
		$xml = $this->doc->saveXML();
		$check = new DOMDocument();
		libxml_use_internal_errors( true );
		$ok = $xml !== false && $check->loadXML( $xml );
		libxml_clear_errors();
		if ( !$ok ) return 'the settings could not be written as valid XML';
		if ( !is_dir( 'backups' ) ) @mkdir( 'backups', 0755, true );
		$backup = 'backups/settings-' . date( 'Ymd' ) . '.xml';
		if ( !file_exists( $backup ) ) @copy( $this->path, $backup );
		$tmp = $this->path . '.tmp-' . getmypid();
		if ( file_put_contents( $tmp, $xml ) === false ) return "{$this->path} could not be written";
		@chmod( $tmp, fileperms( $this->path ) & 0777 );
		if ( !rename( $tmp, $this->path ) ) {
			@unlink( $tmp );
			return "{$this->path} could not be replaced";
		}
		return '';
	}
}
