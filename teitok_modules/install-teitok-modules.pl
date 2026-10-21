#!/usr/bin/env perl
# Install flexipipe's TEITOK pages (teitok_modules/) into a TEITOK *shared* project.
# (Deliberately a copy of flexicorp's install-teitok-ui.pl: each repo installs its own pages.)
#
# TEITOK looks for an action's page in the project's Sources/, then in
# $sharedfolder/Sources/, then in $ttroot/common/Sources/ (and similarly for
# Pages/), so files placed in the shared project work as TEITOK pages without
# touching the TEITOK git checkout:
#
#   *.php  (not *-test.php)   -> SHARED/Sources/
#   *.js, *.css               -> SHARED/Scripts/   (flexipipe.php reads them from there)
#   *.html                    -> SHARED/Pages/     (help pages)
#
# Re-running upgrades in place. A manifest (SHARED/Resources/flexipipe-teitok-modules.json)
# records what was installed, with checksums and the flexipipe version and commit;
# a file that was changed locally since the last install (or that was there before
# the first one) is kept as FILE.local-YYYYmmdd-HHMMSS before it is replaced.
#
# Usage:
#   sudo perl install-teitok-modules.pl [--shared DIR] [--user USER] [--dry-run]
#
#   --shared DIR   the TEITOK shared project (default: TT_SHARED from the TEITOK
#                  .htaccess, else /var/www/html/teitok/shared)
#   --user USER    owner of the installed files (default: owner of DIR)
#   --dry-run      show what would change, change nothing
use strict;
use warnings;
use File::Basename qw(dirname basename);
use File::Copy qw(copy);
use File::Path qw(make_path);
use Cwd qw(abs_path);
use Digest::SHA;
use POSIX qw(strftime);

my ( $shared, $user, $dry ) = ( '', '', 0 );
while ( @ARGV ) {
	my $a = shift @ARGV;
	if    ( $a eq '--shared' )  { $shared = shift @ARGV // ''; }
	elsif ( $a eq '--user' )    { $user = shift @ARGV // ''; }
	elsif ( $a eq '--dry-run' ) { $dry = 1; }
	elsif ( $a eq '-h' || $a eq '--help' ) {
		open my $me, '<', $0 or die; while ( <$me> ) { last if !/^#/; next if /^#!/; s/^# ?//; print; } exit 0;
	} else { die "install-teitok-modules.pl: unknown option $a (see --help)\n"; }
}

my $src = dirname( abs_path($0) );
my $repo = dirname($src);

# ── where is the shared project ─────────────────────────────────────────────
if ( $shared eq '' ) {
	for my $ht ( '/var/www/html/teitok/.htaccess', '/srv/www/htdocs/teitok/.htaccess',
	             '/Library/WebServer/Documents/teitok/.htaccess', '/var/www/localhost/htdocs/teitok/.htaccess' ) {
		next unless -r $ht;
		open my $fh, '<', $ht or next;
		while ( <$fh> ) { if ( /^\s*SetEnv\s+TT_SHARED\s+(\S+)/ ) { $shared = $1; last; } }
		close $fh;
		last if $shared ne '';
	}
	$shared = '/var/www/html/teitok/shared' if $shared eq '';
}
$shared =~ s{/+$}{};
die "install-teitok-modules.pl: $shared is not a directory (pass --shared DIR)\n" unless -d $shared;
die "install-teitok-modules.pl: $shared does not look like a TEITOK project (no Resources/)\n" unless -d "$shared/Resources";

my ( $uid, $gid ) = ( -1, -1 );
if ( $user ne '' ) {
	my @pw = getpwnam($user) or die "install-teitok-modules.pl: no such user $user\n";
	( $uid, $gid ) = @pw[ 2, 3 ];
} else {
	( $uid, $gid ) = ( stat $shared )[ 4, 5 ];
	$user = getpwuid($uid) // $uid;
}

# ── what to install ─────────────────────────────────────────────────────────
opendir my $dh, $src or die "cannot read $src: $!\n";
my @plan;    # [ source file, relative target ]
for my $f ( sort readdir $dh ) {
	next if $f =~ /^\./ || !-f "$src/$f";
	if    ( $f =~ /-test\.php$/ )    { next; }
	elsif ( $f =~ /\.php$/ )         { push @plan, [ $f, "Sources/$f" ]; }
	elsif ( $f =~ /\.(js|css)$/ )    { push @plan, [ $f, "Scripts/$f" ]; }
	elsif ( $f =~ /\.html$/ )        { push @plan, [ $f, "Pages/$f" ]; }
}
closedir $dh;
die "install-teitok-modules.pl: nothing to install in $src\n" unless @plan;

sub sha { my $p = shift; return '' unless -f $p; return Digest::SHA->new(256)->addfile($p, 'b')->hexdigest; }

# version + commit of the flexipipe checkout these files come from
my $version = '';
if ( open my $pp, '<', "$repo/flexipipe/__init__.py" ) {
	while ( <$pp> ) { if ( /^__version__\s*=\s*["']([^"']+)["']/ ) { $version = $1; last; } }
	close $pp;
}
my $commit = '';
if ( -d "$repo/.git" ) {
	$commit = `git -C '$repo' rev-parse --short HEAD 2>/dev/null` // ''; chomp $commit;
	my $dirty = `git -C '$repo' status --porcelain -- '$src' 2>/dev/null` // '';
	$commit .= '-dirty' if $commit ne '' && $dirty ne '';
}

# previous manifest: { "rel": "sha", ... } (tiny JSON, read without a JSON module)
my $manifest = "$shared/Resources/flexipipe-teitok-modules.json";
my %old;
if ( open my $mf, '<', $manifest ) {
	local $/; my $j = <$mf>; close $mf;
	if ( $j =~ /"files"\s*:\s*\{(.*?)\}/s ) {
		my $body = $1;
		while ( $body =~ /"([^"]+)"\s*:\s*"([0-9a-f]{64})"/g ) { $old{$1} = $2; }
	}
}

# ── install ─────────────────────────────────────────────────────────────────
my $stamp = strftime( '%Y%m%d-%H%M%S', localtime );
my ( %new, @changed, @kept_local, $unchanged );
$unchanged = 0;
for my $p ( @plan ) {
	my ( $f, $rel ) = @$p;
	my $from = "$src/$f";
	my $to   = "$shared/$rel";
	my $want = sha($from);
	$new{$rel} = $want;
	my $have = sha($to);
	if ( $have eq $want ) { $unchanged++; next; }
	# changed locally since our last install, or present before the first one: keep a copy
	if ( $have ne '' && ( !exists $old{$rel} || $old{$rel} ne $have ) ) {
		push @kept_local, "$rel -> $rel.local-$stamp";
		copy( $to, "$to.local-$stamp" ) or die "cannot back up $to: $!\n" unless $dry;
	}
	push @changed, ( $have eq '' ? "new      $rel" : "updated  $rel" );
	next if $dry;
	my $dir = dirname($to);
	unless ( -d $dir ) { make_path($dir) or die "cannot create $dir: $!\n"; chown $uid, $gid, $dir; }
	my $tmp = "$to.part-$$";
	copy( $from, $tmp ) or die "cannot write $tmp: $!\n";
	chmod 0644, $tmp;
	chown $uid, $gid, $tmp;
	rename( $tmp, $to ) or die "cannot replace $to: $!\n";
}
my @gone = grep { !exists $new{$_} && -e "$shared/$_" } sort keys %old;

unless ( $dry ) {
	my $tmp = "$manifest.part-$$";
	open my $out, '>', $tmp or die "cannot write $tmp: $!\n";
	print $out "{\n  \"component\": \"flexipipe-teitok-modules\",\n  \"version\": \"$version\",\n",
	           "  \"commit\": \"$commit\",\n  \"source\": \"$src\",\n",
	           "  \"installed\": \"", strftime( '%Y-%m-%dT%H:%M:%S%z', localtime ), "\",\n  \"files\": {\n",
	           join( ",\n", map { "    \"$_\": \"$new{$_}\"" } sort keys %new ), "\n  }\n}\n";
	close $out;
	chmod 0644, $tmp; chown $uid, $gid, $tmp;
	rename( $tmp, $manifest ) or die "cannot replace $manifest: $!\n";
}

# ── copies elsewhere that win over the shared ones ──────────────────────────
# A project's own Sources/ or Scripts/ copy is used before the shared one.
my @shadow;
my $teitokroot = dirname($shared);
if ( opendir my $td, $teitokroot ) {
	for my $proj ( sort readdir $td ) {
		next if $proj =~ /^\./;
		my $pd = "$teitokroot/$proj";
		next if !-d $pd || abs_path($pd) eq abs_path($shared);
		for my $p ( @plan ) {
			push @shadow, "$pd/$p->[1]" if -f "$pd/$p->[1]";
		}
	}
	closedir $td;
}

# ── report ──────────────────────────────────────────────────────────────────
my $what = $dry ? 'would install' : 'installed';
printf "flexipipe TEITOK pages %s %s%s into %s (owner %s)\n", ( $version || '?' ),
	( $commit ne '' ? "($commit) " : '' ), $what, $shared, $user;
print "  $_\n" for @changed;
printf "  %d file(s) already up to date\n", $unchanged if $unchanged;
print "  local changes kept as:\n", map { "    $_\n" } @kept_local if @kept_local;
print "  no longer shipped (left in place): ", join( ', ', @gone ), "\n" if @gone;
if ( @shadow ) {
	print "  warning: these project copies are used instead of the shared ones for their project:\n";
	print "    $_\n" for @shadow;
}
exit 0;
